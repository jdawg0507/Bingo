# Scrabble Engine Specification

## 1. Data Pulling and Dataset Generation
### 1.1 Target Source
- **URL:** [cross-tables.com Annotated Games](https://www.cross-tables.com/annolistself.php)
### 1.2 Objective
- Scrape and parse annotated games from the platform.
- Extract every single play, board state, rack content, and score from all available games.
- Compile these parsed states and moves into a standardized, structured dataset (e.g., CSV, JSON, or specialized binary format) to be used for neural network training.

## 2. Neural Network (NNUE) Training
### 2.1 Architecture Overview
- **Type:** Efficiently Updatable Neural Network (NNUE).
- **Input Representation:**
  - **Board State:** 225 nodes (representing the 15x15 Scrabble board). 
  - **Tile Types:** Up to 27 tile types per square (26 letters + 1 blank tile representation).
  - **Player Rack:** 7 nodes representing the current tiles on the player's rack.
### 2.2 Objective
- Train the NNUE on the aggregated dataset to accurately predict board state values, win probabilities, and expected point spreads.
- Ensure the network is efficiently updatable when only a few pieces on the board change to minimize evaluation overhead during search computations.

## 3. Move Generation and Search
### 3.1 Legal Move Generation
- Implement a fast legal move generator utilizing a word graph structure (such as a GADDAG or DAWG) to find all valid placements on the board.
- Ensure dictionary constraints (e.g., NWL20 or CSW21) and Scrabble rules (cross-checks, anchor squares, bingo bonuses) are strictly enforced.
### 3.2 Evaluation using Monte Carlo
- **Algorithm:** Monte Carlo approaches combined with NNUE evaluation.
- Generate legal moves for the current state.
- Run Monte Carlo simulations to sample possible tile draws from the unseen bag and simulate the opponent's strongest replies.
- Use the updated NNUE to quickly evaluate intermediate and leaf nodes.
- Aggregate simulation results to find the move that maximizes expected value (equity).

## 4. Cleanup and Optimization
### 4.1 Performance Optimization
- Optimize the NNUE inference step utilizing SIMD/vectorization to maximize nodes evaluated per second.
- Implement parallelization for Monte Carlo branches to leverage multi-thread processing.
- Introduce computational optimizations such as transposition tables for caching previously evaluated nodes.
### 4.2 Code Cleanup
- Clean up data-scraping and dataset-formatting scripts.
- modularize the training pipeline and simulation engine.
- Profile the move generator to detect and resolve CPU bottlenecks.
