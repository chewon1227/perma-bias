from typing import List, Dict, Any, Tuple
import random
import math
import fire
from utils import log_function
from careerpathway.treesearch import MCTSVisualizer, CareerState, CareerNode, LLMAgent
from accelerate import infer_auto_device_map
import os
import time
from mylmeval import MyLLMEval, open_json, save_json
import json
from collections import defaultdict
from typing import List, Dict
import re
import graphviz

EXPLORE_NODE_N = {0: 10, 1: 2, 2: 2, 3: 2}

class CareerAction:
    def __init__(self, 
                 action_type: str,
                 details: Dict[str, Any],
                 time_required: int,
                 ):
        self.type = action_type
        self.details = details
        self.time_required = time_required

class CareerAgent:
    def __init__(self, llmagent):
        self.LLMEval = MyLLMEval(model_path=llmagent)
        
    def create_prompt(self, state: CareerState, job_n: int) -> str:
        # 경력 히스토리 수집
        history = []
        current_node = state
        while getattr(current_node, 'parent', None) is not None:
            history.append((current_node.position, current_node.years_experience))
            current_node = current_node.parent
        history.append((current_node.position, current_node.years_experience))
        history.reverse()  # 시간순으로 정렬

        # 경력 히스토리 포맷팅
        career_history = "\n".join([
            f"- {position} ({years} years)"
            for position, years in history
        ])

        prompt = f"""Given the career history below, suggest {job_n} different possible next job positions. Each suggestion should be a different career direction that builds upon this experience.

    Career History:
    {career_history}

    Current Profile:
    - Current Position: {state.position}
    - Years of Experience: {state.years_experience}

    Format your response with one job position per line, exactly like this example:
    Job 1
    Job 2
    Job 3

    Ensure each suggestion is realistic, distinct, and beneficial for career growth, considering the full career history."""

        return prompt

    def _parse_actions(self, text: str, job_n: int) -> List[CareerAction]:
        actions = []
        for line in text.strip().split('\n'):
            if line.strip():
                try:
                    actions.append(CareerAction(
                        'job_change',
                        {'new_position': line.strip()},
                        time_required=60,
                    ))
                except Exception as e:
                    print(f"Error parsing action: {e}")
        return actions[:job_n]
    
    @log_function
    def batch_explore_actions(self, states: List[CareerState], save_path: str, now_depth: int, random_seed: int) -> List[List[CareerAction]]:
        prompts = [self.create_prompt(state, EXPLORE_NODE_N[now_depth]) for state in states]
        
        # Add metadata for tracking
        inference_data = [
            {
                'inputs': [p],
                'metadata': {
                    'graph_id': state.graph_id,
                    'node_id': state.node_id,
                    'stage': 1
                }
            } 
            for p, state in zip(prompts, states)
        ]
        
        responses = self.LLMEval.inference(
            prompt="{}",
            data=inference_data,
            batch_size=len(prompts),
            apply_chat_template=True,
            system_prompt="You are a career advisor suggesting next career moves.",
            save_path=save_path,
            save_additionally=False,
            temperature=0.3,
            max_tokens=1024,
            top_p=0.9,
            top_k=0,
            random_seed=random_seed
        )
        
        all_actions = []
        for response in responses:
            actions = self._parse_actions(response, EXPLORE_NODE_N[now_depth])
            all_actions.append(actions)
            
        return all_actions

class CareerSimulator:
    def __init__(self, action_agent: str, value_agents: List[str], process: str, random_seed: int = 0):
        self.action_agent_name = action_agent
        self.value_agents = value_agents
        self.process = process
        self.all_nodes = []
        self.node_counter = defaultdict(int)
        self.random_seed = random_seed
        
        # Create stage-specific save paths
        self.save_paths = {
            'action': f"/scratch2/iyy1112/results/mcts_value_model/tmp_1_{action_agent.replace('/', '_')}_{self.random_seed}.jsonl",
            'salary': f"/scratch2/iyy1112/results/mcts_value_model/tmp_2_{action_agent.replace('/', '_')}_{self.random_seed}.jsonl",
            'fitness': f"/scratch2/iyy1112/results/mcts_value_model/tmp_3_{action_agent.replace('/', '_')}_{self.random_seed}.jsonl"
        }
        
    def load_previous_stage_data(self, stage: int) -> List[Dict]:
        """Load data from previous stage's output file"""
        prev_stage_path = f"/scratch2/iyy1112/results/mcts_value_model/tmp_{stage-1}_{self.action_agent_name.replace('/', '_')}.jsonl"
        if not os.path.exists(prev_stage_path):
            raise FileNotFoundError(f"Previous stage data not found at {prev_stage_path}")
        return open_json(prev_stage_path)
        
    @staticmethod
    def create_salary_prompt(state: CareerState) -> str:
        return f"Answer the average annual salary for the following job.\n{state.position}"

    @staticmethod
    def create_fitness_prompt(state: CareerState) -> str:
        return f"Answer 1 or 0 for the fitness of the given user information and job name.\nUser information: Not known\nJob name: {state.position}"
    
    def simulate_action_stage(self, data: List[Dict], years: int, step_years: int) -> List[CareerNode]:
        # Initialize states and nodes
        initial_states = []
        initial_nodes = []
        for graph in data:
            for graph_id, items in graph.items():
                initial_node = [r for r in items if r['from']==None][0]['content']
                state = CareerState(
                    current_position=initial_node['main']+" "+initial_node.get('detail', ''),
                    skills=[],
                    years_experience=0,
                    education_level='Bachelor',
                    node_id=0,
                    graph_id=graph_id
                )
                initial_states.append(state)
                node = CareerNode(state)
                initial_nodes.append(node)
                self.all_nodes.append(node)
                
        print(f"Initial states: {len(initial_states)}")
        
        self.action_agent = CareerAgent(self.action_agent_name)
        states_to_explore = [(state, node, 0) for state, node in zip(initial_states, initial_nodes)]
        max_depth = years // step_years
        
        for now_depth in range(max_depth):
            current_states = [state for state, _, _ in states_to_explore]
            all_actions = self.action_agent.batch_explore_actions(
                current_states, 
                save_path=self.save_paths['action'],
                now_depth=now_depth,
                random_seed=self.random_seed
            )
            
            next_states_to_explore = []
            for (state, parent_node, depth), actions in zip(states_to_explore, all_actions):
                for i, action in enumerate(actions[:EXPLORE_NODE_N[now_depth]]):
                    self.node_counter[state.graph_id] += 1
                    new_node_id = self.node_counter[state.graph_id]
                    
                    new_state = self.apply_action(state, action)
                    new_state.node_id = new_node_id
                    new_state.graph_id = state.graph_id
                    
                    child_node = CareerNode(new_state)
                    child_node.action_taken = action
                    child_node.parent = parent_node
                    parent_node.children.append(child_node)
                    
                    self.all_nodes.append(child_node)
                    
                    if depth + 1 < max_depth:
                        next_states_to_explore.append((new_state, child_node, depth + 1))
            
            states_to_explore = next_states_to_explore
            print(f"Depth {now_depth+1}: Created {len(self.all_nodes)} nodes total")
        
        # Save action stage results
        del self.action_agent
        self.save_stage_results(initial_nodes, 1)
        return initial_nodes
    
    def simulate_salary_stage(self) -> None:
        """Stage 2: Salary estimation"""
        # Load previous stage data if needed
        if not self.all_nodes:
            stage1_data = self.load_previous_stage_data(2)
            self.reconstruct_trees(stage1_data)
            
        self.salary_agent = MyLLMEval(self.value_agents[0])
        
        # Prepare batch inference data with metadata
        inference_data = [
            {
                'inputs': [self.create_salary_prompt(node.state)],
                'metadata': {
                    'graph_id': node.state.graph_id,
                    'node_id': node.state.node_id,
                    'stage': 2
                }
            }
            for node in self.all_nodes
        ]
        
        salary_responses = self.salary_agent.inference(
            prompt="{}",
            data=inference_data,
            batch_size=len(inference_data),
            apply_chat_template=True,
            save_path=self.save_paths['salary'],
            save_additionally=True,
            max_tokens=50,
            random_seed=self.random_seed
        )
        
        # Update nodes with salary values
        for node, response in zip(self.all_nodes, salary_responses):
            try:
                salary = float(re.sub(r'[^0-9.]', '', response))
            except:
                salary = None
            if not hasattr(node, 'values'):
                node.values = {}
            node.values['expected_salary'] = salary
            
        self.save_stage_results(self.get_root_nodes(), 2)
        del self.salary_agent
    
    def simulate_fitness_stage(self) -> None:
        """Stage 3: Fitness estimation"""
        # Load previous stage data if needed
        if not self.all_nodes:
            stage2_data = self.load_previous_stage_data(3)
            self.reconstruct_trees(stage2_data)
            
        self.fitness_agent = MyLLMEval(self.value_agents[1])
        
        # Prepare batch inference data with metadata
        inference_data = [
            {
                'inputs': [self.create_fitness_prompt(node.state)],
                'metadata': {
                    'graph_id': node.state.graph_id,
                    'node_id': node.state.node_id,
                    'stage': 3
                }
            }
            for node in self.all_nodes
        ]
        
        fitness_responses = self.fitness_agent.inference(
            prompt="{}",
            data=inference_data,
            batch_size=len(inference_data),
            apply_chat_template=True,
            save_path=self.save_paths['fitness'],
            save_additionally=True,
            max_tokens=50,
            random_seed=self.random_seed
        )
        
        # Update nodes with fitness values
        for node, response in zip(self.all_nodes, fitness_responses):
            try:
                fitness = float(re.sub(r'[^0-9.]', '', response)) * 100
            except:
                fitness = None
            if not hasattr(node, 'values'):
                node.values = {}
            node.values['career_fit'] = fitness
            
        self.save_stage_results(self.get_root_nodes(), 3)
        del self.fitness_agent
    
    def simulate_all_careers(self, data: List[Dict], years: int = 20, step_years: int = 5):
        if self.process == 'action':
            return self.simulate_action_stage(data, years, step_years)
        elif self.process == 'salary':
            self.simulate_salary_stage()
        elif self.process == 'fitness':
            self.simulate_fitness_stage()
        elif self.process == 'all':
            trees = self.simulate_action_stage(data, years, step_years)
            self.simulate_salary_stage()
            self.simulate_fitness_stage()
            return trees
        else:
            raise ValueError(f"Invalid process: {self.process}")
    
    def save_stage_results(self, root_nodes: List[CareerNode], stage: int) -> None:
        """Save stage results in a format that can be reconstructed"""
        results = []
        for tree in root_nodes:
            nodes = list(tree.traverse())
            node_to_idx = {node: idx for idx, node in enumerate(nodes)}
            
            positions = [n.state.position for n in nodes]
            years_experience = [n.state.years_experience for n in nodes]
            values = [getattr(n, 'values', {}) for n in nodes]
            edges = [(node_to_idx[node], [node_to_idx[child] for child in node.children]) for node in nodes]
            
            results.append({
                'graph_id': tree.state.graph_id,
                'initial_node': positions[0],
                'nodes': [
                    {
                        'position': p,
                        'years_experience': y,
                        'values': v,
                        'children_idx': edges[i][1],
                        'node_id': nodes[i].state.node_id
                    }
                    for i, (p, y, v) in enumerate(zip(positions, years_experience, values))
                ]
            })
            
        save_path = f"/scratch2/iyy1112/results/mcts_value_model/tmp_{stage}_{self.action_agent_name.replace('/', '_')}.jsonl"
        save_json(results, save_path)
    
    def reconstruct_trees(self, data: List[Dict]) -> None:
        """Reconstruct career trees from saved data"""
        self.all_nodes = []
        for graph_data in data:
            # Create all nodes first
            nodes = []
            for node_data in graph_data['nodes']:
                state = CareerState(
                    current_position=node_data['position'],
                    skills=[],
                    years_experience=node_data['years_experience'],
                    education_level='Bachelor',
                    node_id=node_data['node_id'],
                    graph_id=graph_data['graph_id']
                )
                node = CareerNode(state)
                if node_data['values']:
                    node.values = node_data['values']
                nodes.append(node)
                self.all_nodes.append(node)
            
            # Connect nodes
            for i, node_data in enumerate(graph_data['nodes']):
                for child_idx in node_data['children_idx']:
                    nodes[i].children.append(nodes[child_idx])
                    nodes[child_idx].parent = nodes[i]
    
    def get_root_nodes(self) -> List[CareerNode]:
        """Get all root nodes from self.all_nodes"""
        return [node for node in self.all_nodes if node.parent is None]
    
    @staticmethod
    def apply_action(state: CareerState, action: CareerAction) -> CareerState:
        new_state = CareerState(
            state.position,
            list(state.skills),
            round(state.years_experience + action.time_required/12),
            state.education_level,
            parent=state  # 부모 상태 설정
        )
        
        if action.type == 'job_change':
            new_state.position = action.details['new_position']
        elif action.type == 'education':
            new_state.education_level = action.details['new_level']
        elif action.type == 'skill_development':
            new_state.skills.add(action.details['new_skill'])
        
        return new_state

def find_best_path(nodes: List[CareerNode], node_to_idx: Dict, a1: float = 1, a2: float = 0) -> List[int]:
    def get_path_score(node: CareerNode) -> Tuple[float, List[int]]:
        if not node.children:
            score = a1 * node.values.get('expected_salary', 0) + a2 * node.values.get('career_fit', 0)
            return score, [node_to_idx[node]]
            
        # 모든 자식 노드들의 경로를 평가
        child_paths = []
        for child in node.children:
            child_score, child_path = get_path_score(child)
            # 현재 노드의 점수를 더함
            total_score = child_score + a1 * node.values.get('expected_salary', 0) + a2 * node.values.get('career_fit', 0)
            child_paths.append((total_score, [node_to_idx[node]] + child_path))
            
        # 가장 높은 점수를 가진 경로 반환
        return max(child_paths, key=lambda x: x[0])

    # 루트 노드부터 시작하여 최적 경로 찾기
    _, best_path = get_path_score(nodes[0])
    return best_path

def main(
    model_path: str = 'Qwen/Qwen2.5-0.5B-Instruct',
    salary_model_path: str = '/scratch2/snail0822/career/job-salary-qwen-3b',
    fitness_model_path: str = '/scratch2/snail0822/career/job-fitness-qwen-0.5b',
    process: str = 'action',  # ['action', 'salary', 'fitness', 'all', 'visualize']
    data_path: str = 'data/evalset/diversity.jsonl',
    years: int = 20,
    step_years: int = 5,
    visualize_sample: int = 5,
    start: int | None = None,
    a1: float = 0.7,
    a2: float = 0.3,
    random_seed: int = 0,
    ):
    
    # Load and prepare data
    data = open_json(data_path)
    graph_data = defaultdict(list)
    for item in data:
        graph_data[item['idx']].append(item)
    
    random.seed(42)
    train_size = int(len(graph_data) * 0.9)
    train_graph_ids = set(random.sample(list(graph_data.keys()), train_size))
    test_graph_ids = set(graph_data.keys()) - train_graph_ids
    test_data = [{idx:graph_data[idx]} for idx in test_graph_ids]
    
    if start:
        test_data = test_data[start:]
    
    # Run simulation
    simulator = CareerSimulator(
        action_agent=model_path,
        value_agents=[salary_model_path, fitness_model_path],
        process=process,
        random_seed=int(random_seed) if random_seed else None
    )
    
    trees = simulator.simulate_all_careers(test_data, years, step_years)
    
    # Only process final results if this is the final stage (fitness) or all stages
    if process in ['fitness', 'visualize', 'all']:
        results = []
        for tree in trees:  # trees는 root node들의 리스트
            # 모든 노드의 정보를 한 번에 수집
            nodes = list(tree.traverse())
            node_to_idx = {node: idx for idx, node in enumerate(nodes)}  # 노드와 인덱스 매핑
            best_path = find_best_path(nodes, node_to_idx, a1=a1, a2=a2)
            
            positions = [n.state.position for n in nodes]
            years_experience = [n.state.years_experience for n in nodes]
            values = [n.values for n in nodes]
            edges = [(node_to_idx[node], [node_to_idx[child] for child in node.children]) for node in nodes]
            
            results.append({
                'graph_id': tree.state.graph_id,
                'initial_node': positions[0],
                'best_path': best_path,
                'nodes': [
                    {
                        'position': p,
                        'years_experience': y,
                        'values': v,
                        'children_idx': edges[i][1],  # 각 노드의 자식 노드들의 인덱스
                        'node_id': nodes[i].state.node_id
                    }
                    for i, (p, y, v) in enumerate(zip(positions, years_experience, values))
                ]
            })

        # Save final results
        final_save_path = f"/scratch2/iyy1112/results/mcts_value_model/{model_path.replace('/', '_')}_{random_seed}.jsonl"
        os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
        save_json(results, final_save_path)
        
        # Visualize only for final stage
        os.makedirs(f"/scratch2/iyy1112/results/mcts_value_model/treemaps/", exist_ok=True)
        visualizer = MCTSVisualizer()
        visualizer.visualize_career_tree(
            data=random.sample(results, min(visualize_sample, len(results))),
            save_path=f"/scratch2/iyy1112/results/mcts_value_model/treemaps/{model_path.replace('/', '_')}_{random_seed}",
        )
    
if __name__ == "__main__": 
    fire.Fire(main)