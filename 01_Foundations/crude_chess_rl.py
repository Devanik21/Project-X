# %% [code]
# advanced_chess_rl.py
# Advanced RL chess agent with UCB exploration, position encoding, and difficulty levels

from collections import defaultdict
import random
import math
import json
import os
import sys
from typing import Dict, List, Tuple

try:
    import chess
    import chess.pgn
except ImportError:
    print("This script needs python-chess. Install with: pip install python-chess")
    sys.exit(1)

# --- Configuration ---
MODEL_FILE = "chess_rl_model.json"
NUM_TRAINING_GAMES = 500
MAX_MOVES_PER_GAME = 300
BASE_TEMPERATURE = 0.8
UCB_C = 1.4  # UCB exploration constant

class ChessRLAgent:
    """Advanced RL agent using Q-learning with position-aware features"""
    
    def __init__(self):
        # Q-values: key = (position_hash, move_uci)
        self.q_values = defaultdict(float)
        # Visit counts for UCB
        self.visit_counts = defaultdict(int)
        self.state_visits = defaultdict(int)
        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.2  # exploration rate during training
        
    def get_state_key(self, board: chess.Board) -> str:
        """Enhanced state representation with material and position info"""
        # Use FEN without move counters for state key
        fen_parts = board.fen().split()
        return f"{fen_parts[0]}_{fen_parts[1]}"
    
    def get_move_features(self, board: chess.Board, move: chess.Move) -> Dict[str, float]:
        """Extract features for a move"""
        features = {}
        
        # Capture bonus
        features['capture'] = 1.0 if board.is_capture(move) else 0.0
        
        # Check bonus
        board.push(move)
        features['check'] = 1.0 if board.is_check() else 0.0
        board.pop()
        
        # Center control (e4, d4, e5, d5)
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        features['center'] = 1.0 if move.to_square in center_squares else 0.0
        
        # Development (moving pieces from back rank early)
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            from_rank = chess.square_rank(move.from_square)
            if (board.turn == chess.WHITE and from_rank == 0) or \
               (board.turn == chess.BLACK and from_rank == 7):
                features['development'] = 1.0
            else:
                features['development'] = 0.0
        else:
            features['development'] = 0.0
            
        return features
    
    def get_q_value(self, board: chess.Board, move: chess.Move) -> float:
        """Get Q-value for a state-action pair"""
        state = self.get_state_key(board)
        key = (state, move.uci())
        return self.q_values[key]
    
    def update_q_value(self, board: chess.Board, move: chess.Move, 
                       reward: float, next_board: chess.Board):
        """Update Q-value using Q-learning update rule"""
        state = self.get_state_key(board)
        key = (state, move.uci())
        
        # Get max Q-value for next state
        if next_board.is_game_over():
            max_next_q = 0.0
        else:
            legal_moves = list(next_board.legal_moves)
            max_next_q = max([self.get_q_value(next_board, m) for m in legal_moves], 
                           default=0.0)
        
        # Q-learning update
        old_q = self.q_values[key]
        self.q_values[key] = old_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - old_q
        )
    
    def select_move_ucb(self, board: chess.Board, level: int = 5) -> chess.Move:
        """Select move using Upper Confidence Bound algorithm"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        state = self.get_state_key(board)
        self.state_visits[state] += 1
        
        # Adjust exploration based on level (lower level = more random)
        exploration_factor = UCB_C * (11 - level) / 5.0
        
        best_score = -float('inf')
        best_moves = []
        
        for move in legal_moves:
            key = (state, move.uci())
            q_val = self.q_values[key]
            visits = self.visit_counts[key]
            
            # UCB score
            if visits == 0:
                ucb_score = float('inf')  # Try unvisited moves first
            else:
                exploration_bonus = exploration_factor * math.sqrt(
                    math.log(self.state_visits[state]) / visits
                )
                ucb_score = q_val + exploration_bonus
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_moves = [move]
            elif ucb_score == best_score:
                best_moves.append(move)
        
        selected = random.choice(best_moves)
        self.visit_counts[(state, selected.uci())] += 1
        return selected
    
    def select_move_level(self, board: chess.Board, level: int) -> chess.Move:
        """Select move based on difficulty level (1-10)"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        if level == 1:
            # Pure random
            return random.choice(legal_moves)
        
        state = self.get_state_key(board)
        
        # Get Q-values for all moves
        move_scores = [(move, self.get_q_value(board, move)) for move in legal_moves]
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Temperature-based selection with level adjustment
        # Higher level = lower temperature = more deterministic
        temperature = BASE_TEMPERATURE * (11 - level) / 5.0
        
        if level >= 9:
            # Almost always pick best move
            if random.random() < 0.95:
                return move_scores[0][0]
        
        # Softmax selection
        scores = [score for _, score in move_scores]
        max_score = max(scores)
        exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]
        
        return random.choices([m for m, _ in move_scores], weights=probs, k=1)[0]
    
    def save_model(self, filename: str):
        """Save model to file"""
        data = {
            'q_values': {f"{k[0]}|{k[1]}": v for k, v in self.q_values.items()},
            'visit_counts': {f"{k[0]}|{k[1]}": v for k, v in self.visit_counts.items()},
            'state_visits': dict(self.state_visits)
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename: str):
        """Load model from file"""
        if not os.path.exists(filename):
            return False
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.q_values = defaultdict(float, {
            tuple(k.split('|')): v for k, v in data['q_values'].items()
        })
        self.visit_counts = defaultdict(int, {
            tuple(k.split('|')): v for k, v in data['visit_counts'].items()
        })
        self.state_visits = defaultdict(int, data['state_visits'])
        print(f"Model loaded from {filename}")
        return True

def train_agent(agent: ChessRLAgent, num_games: int):
    """Train agent through self-play"""
    print(f"\n🎯 Training agent with {num_games} self-play games...")
    
    results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0, '*': 0}
    
    for game_num in range(num_games):
        board = chess.Board()
        game_history = []  # Store (board_state, move) tuples
        
        # Play one game
        for move_num in range(MAX_MOVES_PER_GAME):
            if board.is_game_over():
                break
            
            # Epsilon-greedy exploration during training
            if random.random() < agent.epsilon:
                move = random.choice(list(board.legal_moves))
            else:
                move = agent.select_move_ucb(board, level=7)
            
            game_history.append((board.copy(), move))
            board.push(move)
        
        # Game over - assign rewards
        result = board.result()
        results[result] += 1
        
        # Backpropagate rewards
        if result == '1-0':  # White wins
            final_reward = 1.0
        elif result == '0-1':  # Black wins
            final_reward = -1.0
        elif result == '*':  # Game unfinished (move limit)
            final_reward = 0.0
        else:  # Draw
            final_reward = 0.0
        
        # Update Q-values for all moves in the game
        for i, (board_state, move) in enumerate(game_history):
            # Alternating rewards for white/black
            if board_state.turn == chess.WHITE:
                reward = final_reward
            else:
                reward = -final_reward
            
            # Incremental reward shaping
            if i < len(game_history) - 1:
                next_board = game_history[i + 1][0]
            else:
                next_board = board
            
            agent.update_q_value(board_state, move, reward * 0.1, next_board)
        
        if (game_num + 1) % 50 == 0:
            w, b, d, u = results['1-0'], results['0-1'], results['1/2-1/2'], results['*']
            print(f"  Game {game_num + 1}/{num_games} - "
                  f"W:{w} B:{b} D:{d} Unfinished:{u}")
    
    w, b, d, u = results['1-0'], results['0-1'], results['1/2-1/2'], results['*']
    print(f"\n✅ Training complete! Final: W:{w} B:{b} D:{d} Unfinished:{u}")
    return results

def play_game(agent: ChessRLAgent):
    """Interactive game loop"""
    print("\n" + "="*60)
    print("♟️  CHESS RL AGENT - INTERACTIVE PLAY")
    print("="*60)
    
    # Choose difficulty
    while True:
        try:
            level = int(input("\nSelect difficulty level (1-10, 1=easiest, 10=hardest): "))
            if 1 <= level <= 10:
                break
            print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Choose color
    while True:
        color = input("Play as White or Black? (w/b): ").strip().lower()
        if color in ['w', 'b', 'white', 'black']:
            player_white = color.startswith('w')
            break
        print("Please enter 'w' or 'b'.")
    
    board = chess.Board()
    move_number = 1
    
    print(f"\n🎮 You are playing as {'White' if player_white else 'Black'}")
    print(f"🤖 Agent difficulty: Level {level}/10")
    print("\nCommands: type move in UCI format (e.g., e2e4), 'hint' for suggestion, 'quit' to exit\n")
    
    while not board.is_game_over():
        print(f"\n{'='*60}")
        print(f"Move {move_number}")
        print(f"{'='*60}")
        print(board)
        print()
        
        is_player_turn = (board.turn == chess.WHITE and player_white) or \
                        (board.turn == chess.BLACK and not player_white)
        
        if is_player_turn:
            # Player's turn
            color_name = "white" if board.turn == chess.WHITE else "black"
            while True:
                user_input = input(f"Your move ({color_name}): ").strip().lower()
                
                if user_input in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for playing!")
                    return
                
                if user_input == 'hint':
                    hint_move = agent.select_move_level(board, level=10)
                    print(f"💡 Hint: {hint_move.uci()} (Q={agent.get_q_value(board, hint_move):.3f})")
                    continue
                
                try:
                    move = chess.Move.from_uci(user_input)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("❌ Illegal move. Try again.")
                except:
                    print("❌ Invalid format. Use UCI notation (e.g., e2e4)")
        
        else:
            # Agent's turn
            print(f"🤖 Agent is thinking (level {level})...")
            move = agent.select_move_level(board, level)
            q_val = agent.get_q_value(board, move)
            print(f"🤖 Agent plays: {move.uci()} (Q-value: {q_val:.3f})")
            board.push(move)
        
        if board.turn == chess.WHITE:
            move_number += 1
    
    # Game over
    print(f"\n{'='*60}")
    print("🏁 GAME OVER")
    print(f"{'='*60}")
    print(board)
    print(f"\nResult: {board.result()}")
    
    result = board.result()
    if result == '1/2-1/2':
        print("🤝 It's a draw!")
    elif (result == '1-0' and player_white) or (result == '0-1' and not player_white):
        print("🎉 You win! Congratulations!")
    else:
        print("🤖 Agent wins! Better luck next time!")

def main():
    print("\n" + "="*60)
    print("♟️  ADVANCED CHESS RL AGENT")
    print("="*60)
    
    agent = ChessRLAgent()
    
    # Try to load existing model
    if agent.load_model(MODEL_FILE):
        print("✅ Existing model loaded!")
        action = input("\nTrain more? (y/n): ").strip().lower()
        if action == 'y':
            train_agent(agent, NUM_TRAINING_GAMES)
            agent.save_model(MODEL_FILE)
    else:
        print("🆕 No existing model found. Training new agent...")
        train_agent(agent, NUM_TRAINING_GAMES)
        agent.save_model(MODEL_FILE)
    
    # Play loop
    while True:
        play_game(agent)
        again = input("\n🔄 Play again? (y/n): ").strip().lower()
        if again != 'y':
            break
    
    print("\n👋 Thanks for playing!")

if __name__ == "__main__":
    main()