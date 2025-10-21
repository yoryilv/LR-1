import json
from flask import Flask, request, jsonify
from flask_cors import CORS 
from lr1_parser_generator import generate_lr1_tables, LR1GeneratorError, simulate_parser 

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024  # 16KB max request size
app.config['JSON_SORT_KEYS'] = False  # Preserve order in JSON responses


# --- UTILITY FUNCTIONS ---
def validate_grammar_input(data):
    """Validate grammar input from request."""
    if not data:
        return None, "No JSON data provided"
    
    grammar_text = data.get('grammarText', '').strip()
    if not grammar_text:
        return None, "Grammar text is empty or missing"
    
    # Check for minimum content
    if len(grammar_text) < 5:
        return None, "Grammar text too short (minimum 5 characters)"
    
    # Check for basic grammar format
    if '->' not in grammar_text:
        return None, "Invalid grammar format: missing '->' separator"
    
    start_symbol = data.get('startSymbol', None)
    if start_symbol is not None:
        start_symbol = start_symbol.strip()
        if not start_symbol:
            start_symbol = None
    
    return {'grammar': grammar_text, 'start': start_symbol}, None


def validate_token_input(data):
    """Validate token input from request."""
    if not data:
        return None, "No JSON data provided"
    
    grammar_text = data.get('grammarText', '').strip()
    if not grammar_text:
        return None, "Grammar text is empty or missing"
    
    token_string = data.get('tokenString', '').strip()
    if not token_string:
        return None, "Token string is empty or missing"
    
    start_symbol = data.get('startSymbol', None)
    if start_symbol is not None:
        start_symbol = start_symbol.strip()
        if not start_symbol:
            start_symbol = None
    
    return {
        'grammar': grammar_text, 
        'tokens': token_string, 
        'start': start_symbol
    }, None


def format_table_for_json(table):
    """
    Convert table with int keys to string keys for JSON serialization.
    Handles nested dicts properly.
    """
    if not isinstance(table, dict):
        return table
    
    return {
        str(k): {str(sk): sv for sk, sv in v.items()} if isinstance(v, dict) else v
        for k, v in table.items()
    }


def format_closure_table(closure_data):
    """
    Format closure table data for better frontend rendering.
    Separates kernel and non-kernel items.
    """
    formatted = {}
    
    for state_id, items in closure_data.items():
        formatted[str(state_id)] = {
            'items': items,
            'kernel': [items[0]] if items else [],  # First item is kernel
            'closure': items[1:] if len(items) > 1 else []  # Rest are closure
        }
    
    return formatted


# --- ROOT ENDPOINT ---
@app.route('/', methods=['GET'])
def index():
    """API health check and information."""
    return jsonify({
        "status": "online",
        "service": "LR(1) Parser Generator API",
        "version": "2.0",
        "endpoints": {
            "generate_tables": {
                "path": "/api/generate-tables",
                "method": "POST",
                "description": "Generate LR(1) parsing tables from grammar"
            },
            "parse_tokens": {
                "path": "/api/parse-tokens",
                "method": "POST",
                "description": "Simulate parsing with token string"
            },
            "health": {
                "path": "/api/health",
                "method": "GET",
                "description": "API health check"
            }
        }
    }), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """Detailed health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "LR(1) Parser Generator",
        "timestamp": json.dumps(None)  # Placeholder for timestamp if needed
    }), 200


# --- API Endpoint to Generate LR(1) Tables ---
@app.route('/api/generate-tables', methods=['POST'])
def generate_tables_api():
    """
    Receives grammar input, calculates LR(1) tables, and returns the result as JSON.
    
    Expected JSON payload:
    {
        "grammarText": "S -> A B\nA -> a",
        "startSymbol": "S"  (optional, auto-detected if not provided)
    }
    
    Returns:
    {
        "success": true,
        "data": {
            "table": {...},           // Combined ACTION + GOTO table
            "gotoTable": {...},       // Separate GOTO table
            "actionTable": {...},     // Separate ACTION table (new)
            "rules": [...],           // Production rules with indices
            "allSymbols": [...],      // All grammar symbols
            "terminals": [...],       // Terminal symbols only
            "nonTerminals": [...],    // Non-terminal symbols only
            "numStates": N,           // Number of LR(1) states
            "closureTable": {...},    // Item sets for each state
            "startSymbol": "S"        // Start symbol used
        }
    }
    """
    try:
        # 1. Validate request content type
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Content-Type must be application/json"
            }), 400
        
        # 2. Get and validate input data
        data = request.get_json()
        validated_data, error = validate_grammar_input(data)
        
        if error:
            return jsonify({
                "success": False,
                "error": error
            }), 400
        
        grammar_string = validated_data['grammar']
        start_symbol = validated_data['start']
        
        # 3. Generate LR(1) tables
        # Returns: (combined_table, rules, all_symbols, num_states, 
        #           closure_table_data, goto_table_separate)
        result = generate_lr1_tables(
            grammar_string=grammar_string,
            start_symbol=start_symbol
        )
        
        combined_table, rules, all_symbols, num_states, closure_table_data, goto_table = result
        
        # 4. Process and separate tables
        # Extract ACTION table (terminals only) and GOTO table (non-terminals only)
        action_table = {}
        terminals_set = set()
        non_terminals_set = set()
        
        # Identify terminals and non-terminals from rules
        for rule in rules:
            non_terminals_set.add(rule['nt'])
        
        terminals_set = all_symbols - non_terminals_set - {r['nt'] + "'" for r in rules if r['index'] == 0}
        
        # Separate ACTION from combined table
        for state_id, entries in combined_table.items():
            action_table[state_id] = {}
            for symbol, value in entries.items():
                if symbol in terminals_set or symbol == '$':
                    action_table[state_id][symbol] = value
        
        # 5. Format output for JSON serialization
        response_data = {
            "success": True,
            "data": {
                # Tables
                "table": format_table_for_json(combined_table),
                "gotoTable": format_table_for_json(goto_table),
                "actionTable": format_table_for_json(action_table),
                
                # Grammar information
                "rules": rules,
                "allSymbols": sorted(list(all_symbols)),
                "terminals": sorted(list(terminals_set - {'$'})) + ['$'],
                "nonTerminals": sorted(list(non_terminals_set - {r['nt'] for r in rules if r['index'] == 0})),
                
                # State information
                "numStates": num_states,
                "closureTable": format_closure_table(closure_table_data),
                
                # Metadata
                "startSymbol": rules[0]['prod'][0] if rules else None,  # Original start symbol
                "augmentedStart": rules[0]['nt'] if rules else None
            }
        }
        
        return jsonify(response_data), 200

    except LR1GeneratorError as e:
        # Handle grammar-specific errors (conflicts, invalid rules, etc.)
        return jsonify({
            "success": False,
            "error": str(e),
            "errorType": "GrammarError"
        }), 400
    
    except ValueError as e:
        # Handle value errors (invalid formats, etc.)
        return jsonify({
            "success": False,
            "error": f"Invalid input: {str(e)}",
            "errorType": "ValueError"
        }), 400
    
    except Exception as e:
        # Handle unexpected server errors
        print(f"[ERROR] Unexpected error in generate_tables_api: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}",
            "errorType": "InternalError"
        }), 500


@app.route('/api/parse-tokens', methods=['POST'])
def parse_tokens_api():
    """
    Receives grammar and token string, performs the LR(1) parsing simulation, 
    and returns the step-by-step trace.
    
    Expected JSON payload:
    {
        "grammarText": "S -> A B\nA -> a",
        "tokenString": "a b",
        "startSymbol": "S"  (optional)
    }
    
    Returns:
    {
        "success": true,
        "data": {
            "trace": [...],           // Step-by-step parsing trace
            "result": "Accept",       // Final parsing result
            "steps": N,               // Total number of steps
            "tokens": [...],          // Tokenized input
            "accepted": true          // Boolean result
        }
    }
    """
    try:
        # 1. Validate request content type
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Content-Type must be application/json"
            }), 400
        
        # 2. Get and validate input data
        data = request.get_json()
        validated_data, error = validate_token_input(data)
        
        if error:
            return jsonify({
                "success": False,
                "error": error
            }), 400
        
        grammar_string = validated_data['grammar']
        token_string = validated_data['tokens']
        start_symbol = validated_data['start']
        
        # 3. Tokenize input (split by whitespace)
        tokens = token_string.split()
        
        if not tokens:
            return jsonify({
                "success": False,
                "error": "Token string contains no valid tokens after splitting"
            }), 400
        
        # Validate tokens (basic check)
        for i, token in enumerate(tokens):
            if not token or token.isspace():
                return jsonify({
                    "success": False,
                    "error": f"Invalid empty token at position {i}"
                }), 400
        
        # 4. Call the parser simulation function
        parse_trace = simulate_parser(
            grammar_string=grammar_string,
            tokens=tokens,
            start_symbol=start_symbol
        )
        
        # 5. Analyze results
        if not parse_trace:
            return jsonify({
                "success": False,
                "error": "Parser returned empty trace"
            }), 500
        
        final_step = parse_trace[-1]
        final_status = final_step.get('status', 'Unknown')
        is_accepted = final_status == 'Accept'
        
        # 6. Format response
        response_data = {
            "success": True,
            "data": {
                "trace": parse_trace,
                "result": final_status,
                "steps": len(parse_trace),
                "tokens": tokens,
                "accepted": is_accepted,
                "hasErrors": 'Error' in final_status or 'Fatal' in final_status
            }
        }
        
        return jsonify(response_data), 200

    except LR1GeneratorError as e:
        # Handle grammar-specific errors
        return jsonify({
            "success": False,
            "error": str(e),
            "errorType": "GrammarError"
        }), 400
    
    except ValueError as e:
        # Handle value errors
        return jsonify({
            "success": False,
            "error": f"Invalid input: {str(e)}",
            "errorType": "ValueError"
        }), 400
    
    except RuntimeError as e:
        # Handle runtime errors (like missing GOTO table)
        return jsonify({
            "success": False,
            "error": str(e),
            "errorType": "RuntimeError"
        }), 500
    
    except Exception as e:
        # Handle unexpected errors
        print(f"[ERROR] Simulation error in parse_tokens_api: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "error": f"Internal simulation error: {str(e)}",
            "errorType": "InternalError"
        }), 500


# --- ADDITIONAL UTILITY ENDPOINT ---
@app.route('/api/validate-grammar', methods=['POST'])
def validate_grammar_api():
    """
    Validates grammar syntax without generating tables.
    Useful for quick syntax checking.
    """
    try:
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Content-Type must be application/json"
            }), 400
        
        data = request.get_json()
        validated_data, error = validate_grammar_input(data)
        
        if error:
            return jsonify({
                "success": False,
                "valid": False,
                "error": error
            }), 400
        
        # Try to parse grammar (won't generate full tables)
        grammar_string = validated_data['grammar']
        start_symbol = validated_data['start']
        
        # This will raise LR1GeneratorError if grammar is invalid
        from lr1_parser_generator import LR1Generator
        generator = LR1Generator(grammar_string, start_symbol)
        
        return jsonify({
            "success": True,
            "valid": True,
            "message": "Grammar syntax is valid",
            "rules": len(generator.PRODUCTION_RULES),
            "terminals": len(generator.TERMINALS),
            "nonTerminals": len(generator.NON_TERMINALS)
        }), 200
        
    except LR1GeneratorError as e:
        return jsonify({
            "success": False,
            "valid": False,
            "error": str(e),
            "errorType": "GrammarError"
        }), 400
    
    except Exception as e:
        return jsonify({
            "success": False,
            "valid": False,
            "error": str(e),
            "errorType": "InternalError"
        }), 500


# --- ERROR HANDLERS ---
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "availableEndpoints": ["/api/generate-tables", "/api/parse-tokens", "/api/health"]
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        "success": False,
        "error": "Method not allowed. Check API documentation for correct HTTP method."
    }), 405


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle 413 errors (payload too large)."""
    return jsonify({
        "success": False,
        "error": "Request payload too large (maximum 16KB)"
    }), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    print(f"[ERROR] Internal server error: {error}")
    return jsonify({
        "success": False,
        "error": "Internal server error. Please check server logs."
    }), 500


@app.errorhandler(Exception)
def handle_exception(error):
    """Global exception handler."""
    print(f"[ERROR] Unhandled exception: {error}")
    import traceback
    traceback.print_exc()
    
    return jsonify({
        "success": False,
        "error": "An unexpected error occurred",
        "errorType": type(error).__name__
    }), 500


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    print("=" * 70)
    print("  LR(1) Parser Generator API - Enhanced Version")
    print("=" * 70)
    print(f"  Server: http://127.0.0.1:5000")
    print(f"  Status: http://127.0.0.1:5000/")
    print(f"  Health: http://127.0.0.1:5000/api/health")
    print()
    print("  Endpoints:")
    print("    POST /api/generate-tables  - Generate LR(1) parsing tables")
    print("    POST /api/parse-tokens     - Simulate parsing with tokens")
    print("    POST /api/validate-grammar - Validate grammar syntax")
    print("    GET  /api/health           - Health check")
    print("    GET  /                     - API information")
    print()
    print("=" * 70)
    print()
    
    # Run the Flask application
    app.run(
        debug=True,
        host='127.0.0.1',
        port=5000,
        threaded=True  # Allow multiple concurrent requests
    )