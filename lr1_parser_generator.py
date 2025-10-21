import json
from collections import defaultdict

# --- 1. DS ---

class LR1Parser:
    """Simulates the LR(1) parser using generated tables."""

    def __init__(self, combined_table, production_rules, goto_table=None):
        self.combined_table = combined_table # ACTION + GOTO 
        self.production_rules = production_rules
        
        if not goto_table:
            raise RuntimeError("GOTO table is required for LR(1) parser initialization")
        
        self.goto_table = goto_table

    def get_action(self, state, token):
        return self.combined_table.get(state, {}).get(token)

    def get_goto(self, state, nonterminal):
        return self.goto_table.get(state, {}).get(nonterminal)

    def simulate(self, input_tokens):
        trace = []
        token_index = 0
        state_stack = [0]
        symbol_stack = []
        error_count = 0
        max_errors = 5

        while True:
            current_state = state_stack[-1]
            current_token = input_tokens[token_index] if token_index < len(input_tokens) else '$'
            action = self.get_action(current_state, current_token)

            stack_display = ' '.join([f"{sym}{state}" for sym, state in zip(symbol_stack, state_stack[1:])])
            if not stack_display:
                stack_display = str(state_stack[0])
            else:
                stack_display = str(state_stack[0]) + ' ' + stack_display

            step_data = {
                'step': len(trace),
                'stack': stack_display,
                'input': ' '.join(input_tokens[token_index:]),
                'action': action or 'error',
                'status': 'Processing'
            }
            trace.append(step_data)

            if not action:
                error_count += 1
                step_data['status'] = f'Error: No action defined for state {current_state} with token "{current_token}"'
                
                if error_count >= max_errors:
                    step_data['status'] = 'Fatal: Too many errors'
                    break
                
                if token_index < len(input_tokens) - 1:
                    token_index += 1
                    step_data['recovery'] = f'Skipped token "{current_token}", moved to next'
                else:
                    step_data['status'] = 'Fatal: Unexpected end of input'
                    break
                continue

            if action.startswith('s'):
                # SHIFT action
                next_state = int(action[1:])
                state_stack.append(next_state)
                symbol_stack.append(current_token)
                token_index += 1
                step_data['action'] = f"Shift to state {next_state}"

            elif action.startswith('r'):
                # REDUCE action
                rule_index = int(action[1:])
                
                if rule_index >= len(self.production_rules):
                    step_data['status'] = f'Fatal: Invalid rule index {rule_index}'
                    break
                
                rule = self.production_rules[rule_index]
                lhs = rule['nt']
                rhs_len = len(rule['prod']) if rule['prod'] != ['epsilon'] else 0

                # (remove rhs_len symbols and states)
                if rhs_len > 0:
                    state_stack = state_stack[:-rhs_len]
                    symbol_stack = symbol_stack[:-rhs_len]

                # explicit use of GOTO table
                goto_state = self.get_goto(state_stack[-1], lhs)
                
                if goto_state is None:
                    step_data['status'] = f'Fatal: GOTO({state_stack[-1]}, {lhs}) undefined - Grammar has conflicts'
                    break

                if not isinstance(goto_state, int):
                    step_data['status'] = f'Fatal: GOTO returned invalid type {type(goto_state).__name__} instead of int'
                    break

                state_stack.append(goto_state)
                symbol_stack.append(lhs)
                
                prod_str = ' '.join(rule['prod']) if rule['prod'] else 'ε'
                step_data['action'] = f"Reduce by R{rule_index}: {lhs} → {prod_str}"

            elif action == 'acc':
                # ACCEPT action
                step_data['status'] = 'Accept'
                step_data['action'] = 'Accept'
                break

            # prevent infinite loops
            if len(trace) > 1000:
                step_data['status'] = 'Fatal: Execution limit exceeded (possible infinite loop)'
                break

        return trace

## Throw punc errors while parsing 
class LR1GeneratorError(Exception):
    pass

## LR1-ITEM UNDERSTOOD
class LR1Item:
    """Represents a single LR(1) item (or configuration)."""
    def __init__(self, nt, production, dot_index, lookahead):
        self.nt = nt # LIKE LHS
        self.production = production # LIKE RHS
        self.dot_index = dot_index # DOT IDX 
        self.lookahead = lookahead # AFTER THE COMMA

    def is_reducible(self):
        return self.dot_index == len(self.production) ## F v T

    def next_symbol(self):
        if self.is_reducible():
            return None # is@end
        return self.production[self.dot_index] 

    def __eq__(self, other):
        return (self.nt == other.nt and
                self.production == other.production and
                self.dot_index == other.dot_index and
                self.lookahead == other.lookahead) #saves space (for duplicates)

    def __hash__(self):
        return hash((self.nt, tuple(self.production), self.dot_index, self.lookahead)) # idx of prod

    def __repr__(self):
        prod_str = [*self.production[:self.dot_index], '•', *self.production[self.dot_index:]] # how we visually see it
        return f"[{self.nt} → {' '.join(prod_str)}, {self.lookahead}]"


# --- 2. Grammar Parsing and LR(1) Engine Class ---

class LR1Generator:
    """Encapsulates the entire LR(1) table generation process."""

    def __init__(self, raw_grammar_string, start_symbol_override=None):
        self.raw_grammar_string = raw_grammar_string
        self.start_symbol = start_symbol_override
        
        self.PRODUCTION_RULES = []
        self.NON_TERMINALS = set()
        self.TERMINALS = set()
        self.FIRST_SETS = defaultdict(set)
        
        self._parse_grammar()
        self._compute_first_sets()

    def _parse_grammar(self):
        lines = self.raw_grammar_string.split('\n')
        all_symbols = set()
        rule_index = 0
        current_start_symbol = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split('->', 1)
                if len(parts) != 2:
                    raise LR1GeneratorError(f"Invalid grammar format (missing '->'): {line}")
                
                nt, prod_list_str = [p.strip() for p in parts]
                
                if not nt:
                    raise LR1GeneratorError(f"Non-terminal cannot be empty in rule: {line}")
                
                if not current_start_symbol:
                    current_start_symbol = nt

                self.NON_TERMINALS.add(nt)
                
                for prod_str in prod_list_str.split('|'):
                    prod_symbols = prod_str.split()

                    if not prod_symbols or prod_symbols == ["''"]: ## EPSILON ERR FIX
                        prod_symbols = ['epsilon']
                    
                    # Validate prod before adding
                    self._validate_production(nt, prod_symbols, line)
                    
                    self.PRODUCTION_RULES.append({
                        'index': rule_index + 1,
                        'nt': nt,
                        'prod': prod_symbols
                    })
                    rule_index += 1
                    all_symbols.update(prod_symbols)
            
            except LR1GeneratorError:
                raise
            except Exception as e:
                raise LR1GeneratorError(f"Error parsing rule '{line}': {e}")

        if not self.PRODUCTION_RULES:
            raise LR1GeneratorError("No valid grammar rules found")

        self.start_symbol = self.start_symbol or current_start_symbol
        
        # aug grammar
        augmented_nt = f"{self.start_symbol}'"
        self.PRODUCTION_RULES.insert(0, {
            'index': 0,
            'nt': augmented_nt,
            'prod': [self.start_symbol]
        })
        self.NON_TERMINALS.add(augmented_nt)
        
        # det terminals
        self.TERMINALS = (all_symbols - self.NON_TERMINALS) | {'$', 'epsilon'}

    def _validate_production(self, nt, prod_symbols, original_line):
        
        if all(s.strip() == '' for s in prod_symbols):
            raise LR1GeneratorError(
                f"Production contains only whitespace in rule: {original_line}"
            )
        
        for i, sym in enumerate(prod_symbols):
            if not sym or sym.isspace():
                raise LR1GeneratorError(
                    f"Empty or whitespace symbol at position {i} in rule: {original_line}"
                )
        
        return True

    def _get_production_index(self, nt, rhs):
        search_rhs = rhs if rhs else ['epsilon']
        return next((rule['index'] for rule in self.PRODUCTION_RULES 
                    if rule['nt'] == nt and rule['prod'] == search_rhs), -1)
    

    def _compute_first_set(self, symbol_list):
        first_set = set()
        
        for symbol in symbol_list:
            #get FIRST set del actual_sym
            symbol_first = self.FIRST_SETS[symbol]
            
            # add everything except epsilon ## RULE
            first_set.update(symbol_first - {'epsilon'})
            
            # Si 'epsilon' no está en el FIRST del sym actual, detenemos la propagación
            if 'epsilon' not in symbol_first:
                return first_set # Devolvemos el conjunto w/o epsilon
                
        first_set.add('epsilon')
        return first_set

    def _compute_first_sets(self):
        # Inicializar con 'epsilon' como un terminal que solo se contiene a sí mismo.
        self.FIRST_SETS = {sym: {sym} for sym in self.TERMINALS} | {nt: set() for nt in self.NON_TERMINALS}
        self.FIRST_SETS['epsilon'] = {'epsilon'}

        changed = True
        while changed:
            changed = False
            for rule in self.PRODUCTION_RULES:
                # Si es A -> '', FIRST(A) debe incluir ''.
                if rule['prod'] == ['epsilon']:
                    old_size = len(self.FIRST_SETS[rule['nt']])
                    self.FIRST_SETS[rule['nt']].add('epsilon')
                    if len(self.FIRST_SETS[rule['nt']]) != old_size:
                        changed = True
                    continue

                # Para otras reglas, calcular el first set de la producción
                temp_first = self._compute_first_set(rule['prod'])
                
                old_size = len(self.FIRST_SETS[rule['nt']])
                self.FIRST_SETS[rule['nt']].update(temp_first)
                
                if len(self.FIRST_SETS[rule['nt']]) != old_size:
                    changed = True

    def _compute_closure(self, item_set):
        closure_set = set(item_set)
        changed = True

        while changed:
            changed = False
            new_items = set()

            for item in closure_set:
                X = item.next_symbol()
                if X not in self.NON_TERMINALS:
                    continue
                    
                beta_a = item.production[item.dot_index + 1:] + [item.lookahead]
                
                first_of_beta_a = self._compute_first_set(beta_a)
                
                for rule in self.PRODUCTION_RULES:
                    if rule['nt'] == X:
                        for b in first_of_beta_a:
                            # Si la producción es X -> epsilon, el dot_index es 0
                            prod_body = [] if rule['prod'] == ['epsilon'] else rule['prod']
                            new_item = LR1Item(X, prod_body, 0, b) # Usar prod_body
                            
                            if new_item not in closure_set and new_item not in new_items:
                                new_items.add(new_item)
                                changed = True
            
            closure_set.update(new_items)
            
        return frozenset(closure_set)

    def _compute_goto(self, item_set, symbol):
        goto_set = {LR1Item(item.nt, item.production, item.dot_index + 1, item.lookahead)
                    for item in item_set if item.next_symbol() == symbol}
        return self._compute_closure(goto_set) if goto_set else frozenset()

    def _build_lr1_states(self):
        augmented_start = f"{self.start_symbol}'"
        I0 = self._compute_closure({LR1Item(augmented_start, [self.start_symbol], 0, '$')})
        
        states = [I0]
        state_map = {I0: 0}
        goto_map = {}
        symbols_to_check = (self.TERMINALS | self.NON_TERMINALS) - {'$'}
        queue = [I0]

        while queue:
            current_state = queue.pop(0)
            i = state_map[current_state]

            for X in symbols_to_check:
                next_state = self._compute_goto(current_state, X)
                if not next_state:
                    continue
                    
                if next_state not in state_map:
                    j = len(states)
                    states.append(next_state)
                    state_map[next_state] = j
                    queue.append(next_state)
                else:
                    j = state_map[next_state]
                
                goto_map[(i, X)] = j
        
        return states, goto_map

    def _build_parsing_tables(self, states, goto_map):
        action_table = defaultdict(dict)
        goto_table = defaultdict(dict)
        terminals_with_dollar = self.TERMINALS
        augmented_start = f"{self.start_symbol}'"
        
        for i, state in enumerate(states):
            # Usamos el goto_map que ya resume las transiciones sin duplicados.
            for symbol in self.TERMINALS.union(self.NON_TERMINALS):
                target_state_idx = goto_map.get((i, symbol))
                if target_state_idx is None:
                    continue

                if symbol in self.NON_TERMINALS:
                    # GOTO para non-terms
                    if symbol != augmented_start:
                        goto_table[i][symbol] = target_state_idx
                elif symbol in terminals_with_dollar:
                    # desplazar para terminals
                    action_table[i][symbol] = f's{target_state_idx}'

            
            for item in state:
                if item.is_reducible():
                    rule_idx = self._get_production_index(item.nt, item.production)
                    lookahead = item.lookahead
                    
                    # Accept Action
                    if rule_idx == 0 and lookahead == '$':
                        action_table[i]['$'] = 'acc'
                    # Reduce Action
                    elif rule_idx > 0:
                        if lookahead in action_table[i]:
                            existing_action = action_table[i][lookahead]
                            if existing_action.startswith('s'):
                                raise LR1GeneratorError(
                                    f"Shift/Reduce conflict in state I{i} on '{lookahead}' (S{existing_action[1:]} vs R{rule_idx})"
                                )
                            if existing_action.startswith('r'):
                                # Solo es un error si es una regla diferente
                                if existing_action != f'r{rule_idx}':
                                    raise LR1GeneratorError(
                                        f"Reduce/Reduce conflict in state I{i} on '{lookahead}' (R{rule_idx} vs R{existing_action[1:]})"
                                    )
                        else:
                            action_table[i][lookahead] = f'r{rule_idx}'
                            
        return action_table, goto_table

    def generate_combined_table(self):
        states, goto_map = self._build_lr1_states()
        action_table, goto_table = self._build_parsing_tables(states, goto_map)
        
        combined_table = {i: {**action_table[i], **goto_table[i]} 
                        for i in range(len(states))}
        
        goto_table_separate = dict(goto_table)
        
        # Closure table para la vista ja
        closure_table_data = {
            i: [str(item) for item in state] for i, state in enumerate(states)
        }

        return (combined_table, self.PRODUCTION_RULES, 
                self.TERMINALS.union(self.NON_TERMINALS), len(states), 
                closure_table_data, goto_table_separate)


# --- 3. Main Execution ---

def generate_lr1_tables(grammar_string, start_symbol):
    generator = LR1Generator(grammar_string, start_symbol)
    return generator.generate_combined_table()  # Returns 6 values


def simulate_parser(grammar_string, tokens, start_symbol):
    tokens_copy = [t for t in tokens if t != '$']
    tokens_copy.append('$')

    combined_table, rules, _, _, _, goto_table = generate_lr1_tables(grammar_string, start_symbol)
    
    parser = LR1Parser(combined_table, rules, goto_table=goto_table)
    
    trace = parser.simulate(tokens_copy) # Execute simulation
    
    return trace


if __name__ == '__main__':
    grammar = "S -> C C\nC -> c C | d"
    try:
        combined_table, rules, symbols, states, closure_table, goto_table = generate_lr1_tables(grammar, 'S')
        print(f"Generated {states} LR(1) states from {len(rules)} rules")
        print(json.dumps(dict(list(combined_table.items())[:3]), indent=2))
        
        print("\nTesting parser simulation:")
        trace = simulate_parser(grammar, ['c', 'd', 'c', 'd'], 'S')
        print(f"Parse result: {trace[-1]['status']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()