import pandas as pd
from environment import ELTEnvironment

def run_e2e_rl_simulation():
    print("=== Starting E2E RL Training Loop Simulation ===")
    
    # setup mock data
    source_df = pd.DataFrame({
        'id': range(1, 6),
        'category': ['A', 'B', 'A', 'C', 'B'],
        'amount': [10, 20, 30, 40, 50]
    })
    
    golden_df = source_df.groupby('category', as_index=False)['amount'].sum()
    
    env = ELTEnvironment(target_table_name="target_aggregate")
    
    # simulate a few policy iterations: bad syntax -> wrong data -> correct logic
    agent_attempts = [
        # try 1: random garbage code
        "import duckdb\nconn = duckdb.connect(os.environ['DUCKDB_PATH'])\nBROKEN SYNTAX HERE",
        
        # try 2: valid syntax, but wrong table/values
        """
import os
import duckdb
import pandas as pd

db_path = os.environ['DUCKDB_PATH']
target_table = os.environ['TARGET_TABLE']

conn = duckdb.connect(db_path)
conn.execute(f"CREATE TABLE {target_table} AS SELECT 'wrong' as category, 0 as amount")
conn.close()
        """,
        
        # try 3: actual correct groupby sum query
        """
import os
import duckdb
import pandas as pd

db_path = os.environ['DUCKDB_PATH']
source_table = os.environ['SOURCE_TABLE']
target_table = os.environ['TARGET_TABLE']

conn = duckdb.connect(db_path)
query = f'''
    CREATE TABLE {target_table} AS 
    SELECT category, SUM(amount) AS amount 
    FROM {source_table} 
    GROUP BY category
'''
conn.execute(query)
conn.close()
        """
    ]
    
    episode_rewards = []
    
    for episode, action_code in enumerate(agent_attempts, 1):
        print(f"\n--- Episode {episode} ---")
        
        obs = env.reset(source_df, golden_df)
        print(f"Environment reset. DB path created: {obs['db_path']}")
        
        reward, done, info = env.step(action_code)
        episode_rewards.append(reward)
        
        print(f"Action Success: {info['success']}")
        print(f"Reward Received: {reward}")
        if info['stderr']:
            print(f"Stderr captured: {info['stderr'].strip()}")

    print("\n=== E2E Simulation Summary ===")
    print(f"Episode Rewards Trajectory: {episode_rewards}")
    
    assert episode_rewards == [0.0, 0.0, 1.0], f"Expected [0.0, 0.0, 1.0], got {episode_rewards}"
    print("SUCCESS: E2E RL training loop simulation verified. Rewards successfully increased as policy improved!")

if __name__ == "__main__":
    run_e2e_rl_simulation()