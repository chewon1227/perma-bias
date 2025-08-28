from h1_lexical import TreeRetrieval
from typing import List, Tuple, Dict
from mylmeval import open_json, save_json
from mylmeval.llm import MyLLMEval
from careerpathway.utils import get_random_name
from tqdm import tqdm
from scipy.spatial.distance import cdist
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import random
import torch
import os
import gc

def em_multi_multi(
    queries: List[List[str]],
    documents: List[List[str]],
    top_k: int = 5
):
    # return top_k results wtih num of intersection
    results = []
    for query_list in tqdm(queries):
        retrieved = []
        for doc_i, document_list in enumerate(documents):
            intersection = len(set(query_list).intersection(set(document_list)))
            retrieved.append((doc_i, intersection, document_list))
        sorted_results = sorted(retrieved, key=lambda x: x[1], reverse=True)[:top_k]
        results.append(sorted_results)
    return results


class Tree:
    def __init__(self, 
                batch_size: int = 32,
                model_name_or_path: str = 'Qwen/Qwen2.5-0.5B-Instruct',
                top_k_list: List[int] = [10, 2, 2, 2],
                method: str | List = 'gen',
                documents=None, # ret
                data=None, # ret
                language='en', # gen
                prompt=None, # gen
                parsing_function=None, # gen
                save_dir='results/3_tree',
                ret_type='lexical',
                temperature=0.7,
                threshold=0.5,
                beam_size=None,
                embedding_model='paraphrase-multilingual-mpnet-base-v2',
                do_bias=False,
                save_tmp=False
                ):
        self.batch_size = batch_size
        self.methods = method
        self.model_name_or_path = model_name_or_path
        self.top_k_list = top_k_list
        self.save_dir = save_dir
        self.ret_type = ret_type
        self.llmeval = MyLLMEval(model_path=self.model_name_or_path)
        self.temperature = temperature
        self.threshold = threshold
        self.beam_size = beam_size
        self.do_bias=do_bias
        self.save_tmp = save_tmp
        self.retrieve_lookback = False
        
        if len(self.top_k_list) != len(self.methods):
            raise ValueError("Length of top_k_list and method should be the same")
        
        if 'retrieve' in self.methods or 'ret' in self.methods or 'pivot_gen' in self.methods:
            self.retriever = TreeRetrieval(batch_size=self.batch_size)
            self.data = data
            self.documents = documents
            if self.data == None:
                self.data = [r for r in open_json("data/data13_15_kaggle.jsonl") if r['skills'] != None]
                self.documents = [r['skills'] for r in self.data]

        if 'gen' in self.methods or 'pivot_gen' in self.methods:
            self.language = language
            self.prompt = prompt
            self.parsing_function = parsing_function
            
        if beam_size:
            self.sentence_model = SentenceTransformer(embedding_model)
            
    def _get_new_node_id(self, tree_id: int) -> int:
        return len([r for r in self.trees if r['graph_id'] == tree_id][0]['nodes'])
    
    
    def _get_parents(self, query_node) -> List[Dict]:
        parents = []
        graph_id = query_node['graph_id']
        all_nodes = [r for r in self.trees if r['graph_id'] == graph_id][0]['nodes']
        for parent_id in query_node['parent_id'] + [query_node['node_id']]:
            parent = [r for r in all_nodes if r['node_id'] == parent_id][0]
            parents.append(parent)
        return parents
        
    
    def _get_siblings(self, query_node) -> List[str]:
        graph_id = query_node['graph_id']
        stage = query_node['stage']
        all_nodes = [r for r in self.trees if r['graph_id'] == graph_id][0]['nodes']
        siblings = [r for r in all_nodes if stage+1 == r['stage']]
        print(f"Stage: {stage+1}, Siblings: {siblings}")
        return siblings
    
    
    def _get_prompt(self, query_node: Dict, job_n:int, bias: str | None) -> str:
        history = self._get_parents(query_node)
        requirements = list(set([r['requirements'] for r in history if 'requirements' in r and r['requirements'] is not None]))
        if len(requirements) > 0:
            current_skills = ', '.join(requirements)
        else:
            current_skills = 'Not Available'
        prompt = self.prompt.format(
            job_n, 
            ', '.join([f"{r['content']} ({r['year'] if 'year' in r else 5} years)" for r in history]), 
            query_node['content'],
            sum([r['year'] if 'year' in r and r['year'] != None else 5 for r in history]),
            current_skills
            )
        if bias:
            random_name = get_random_name(bias.split("_")[0], bias.split("_")[1]) # nation, sex
            prompt = prompt.replace('\n\nFormat your response as follows for each position:', f'\n- Name: {random_name}\n\nFormat your response as follows for each position:')
        return prompt
    
    
    def tree_update(self, new_node: Dict):
        graph_id = new_node['graph_id']
        for tree in self.trees:
            if tree['graph_id'] == graph_id:
                tree['nodes'].append(new_node)
                return
                
                        
    def load(self, queries: List[dict], top_k: int, stage: int, load_where: str) -> List[List[str]]:
        new_queries = []
        whole_generations = open_json(load_where.format(self.model_name_or_path.replace("/", "_"))+'.jsonl')
        if len(whole_generations) != len(queries):
            raise ValueError(f"Length of whole_generations({len(whole_generations)}) and queries({len(queries)}) should be the same")
        for i, (item, query) in enumerate(zip(whole_generations, queries)):
            top_k_results = self.parsing_function(item['result'], top_k)[:top_k]
            for id, r in enumerate(top_k_results):
                new_node = {'content' : r, 'stage' : stage, 'parent_id' : query['parent_id'] + [query['node_id']], 'type' : 'gen', 'node_id' : self._get_new_node_id(tree_id=query['graph_id']), 'graph_id': query['graph_id']}
                self.tree_update(new_node)
                new_queries.append(new_node)
        return new_queries
    
    
    def annotate_skills(self, queries):
        generations = [
            "Business Administration, Economics, Marketing, Danish Language, International Trade, Business Communication, Financial Management, Business Mathematics, Project Management",
            "Software Development, Programming Languages, Database Management, Version Control, Agile Methodologies, Software Testing, Problem-Solving, Technical Documentation, Team Collaboration",
            "Strategic Communication, Public Relations, Crisis Management, Media Relations, Communication Strategy, Organizational Communication, Research Methods, Project Management, Leadership, Presentation Skills",
            "Training Development, Instructional Design, Public Speaking, Adult Learning Principles, Performance Assessment, Training Program Management, Workshop Facilitation, E-learning Development, Communication Skills, Training Evaluation",
            "Business Strategy, Financial Analysis, Management Theory, Marketing Management, Operations Management, Business Research, Leadership Development, Strategic Planning, Data Analysis, Decision Making",
            "Business Development, Technical Sales, Market Analysis, Product Knowledge, Client Relations, Proposal Writing, Industrial Engineering Principles, Project Coordination, Technical Documentation, Customer Needs Assessment",
            "Cross-cultural Communication, International Marketing, Business Writing, Intercultural Management, Global Business Strategy, Corporate Communication, Language Skills, Stakeholder Management, International Relations, Marketing Communication",
            "Account Management, Sales Strategy, Client Relationship Management, Technology Products Knowledge, Sales Forecasting, Pipeline Management, Negotiation Skills, Business Development, Customer Service, Solution Selling",
            "Media Production, Content Creation, Digital Media, Journalism, Social Media Management, Media Analysis, Communication Theory, Media Writing, Audio/Video Production, Media Planning",
            "Software Engineering, Programming Languages, System Design, Database Management, API Development, Code Testing, Debugging, Agile Development, Technical Problem-Solving, Software Architecture",
            "Quality Management Systems, ISO 9001 Standards, Internal Auditing, Process Improvement, Documentation Review, Compliance Assessment, Risk Management, Quality Control, Corrective Action Planning, Audit Reporting",
            "Financial Accounting, Business Finance, Cost Accounting, Taxation, Financial Analysis, Business Law, Corporate Finance, Financial Management, Business Mathematics, Economics",
            "Customer Service Management, Client Relations, Service Coordination, Account Management, Problem Resolution, Communication Skills, Administrative Support, Client Documentation, Service Quality Monitoring, Team Coordination",
            "Robotics Programming, Automation Systems, Mechanical Design, Electronic Systems, Control Systems, PLC Programming, CAD/CAM, Sensor Integration, Industrial Automation, System Integration",
            "Digital Business Strategy, E-commerce Analytics, Online Marketing, Digital Economics, Payment Systems, Supply Chain Management, Web Analytics, Digital Product Management, E-business Models, Market Analysis",
            "Not known",
            "Software Testing, Test Automation, Quality Assurance, Bug Tracking, Test Case Development, Regression Testing, Performance Testing, Test Documentation, Testing Methodologies, Defect Management",
            "International Business Strategy, Global Marketing, Cross-cultural Management, International Trade, Business Development, Strategic Planning, Financial Management, Global Operations Management, Leadership Skills, Market Research",
            "Aerodynamics, Aircraft Design, Propulsion Systems, Flight Mechanics, Structural Analysis, Thermodynamics, Engineering Mathematics, CAD Software, Fluid Dynamics, Space Systems",
            "Historical Research, Document Analysis, Critical Thinking, Academic Writing, Research Methodology, Historical Analysis, Source Evaluation, Analytical Skills, Historical Documentation, Academic Research",
            "Process Engineering, Chemical Process Design, Thermodynamics, Heat Transfer, Mass Transfer, Reaction Engineering, Process Control, Plant Design, Material Science, Chemical Laboratory Techniques",
            "Programming Languages, Data Structures, Algorithms, Database Systems, Software Engineering, Operating Systems, Computer Networks, System Architecture, Object-Oriented Programming, Web Development",
            "Financial Management, Marketing Strategy, Market Analysis, Investment Analysis, Portfolio Management, Business Strategy, Financial Planning, Brand Management, Marketing Research, Risk Management",
            "Investment Operations, Portfolio Management, Risk Assessment, Financial Analysis, Asset Management, Compliance Monitoring, Trade Settlement, Performance Reporting, Investment Strategy, Team Management",
            "Not known",
            "Java Programming, Object-Oriented Programming, Software Development, Application Design, Database Integration, Web Applications, API Development, Testing and Debugging, Version Control, Software Documentation",
            "Circuit Design, Power Systems, Electronic Systems, Signal Processing, Control Systems, Digital Electronics, Electromagnetic Theory, Communication Systems, Microprocessor Systems, Engineering Mathematics",
            "Optical System Design, Laser Technology, Photonics, Optical Communication, Image Processing, Optical Sensors, Fiber Optics, Optoelectronics, Wave Optics, Optical Materials",
            "Project Management Methodology, Project Planning, Risk Management, Project Control, Change Management, Quality Management, Project Organization, Business Case Development, Project Governance, Stakeholder Management",
            "Financial Analysis, Corporate Finance, Investment Management, Financial Markets, Business Strategy, Accounting Principles, Economics, Business Law, Risk Assessment, Financial Planning",
            "Public Relations Strategy, Marketing Communications, Business Foundation, Media Relations, Campaign Planning, Brand Management, Business Communication, Social Media Management, Crisis Communication, Market Research",
            "UI Design, User Experience, Interface Prototyping, Usability Testing, Interaction Design, Information Architecture, Visual Design, User Research, Interface Evaluation, Design Principles",
            "Physics, Chemistry, Mathematics, Biology, Scientific Methods, Laboratory Techniques, Data Analysis, Problem Solving, Scientific Writing, Research Methods",
            "Technical Drawing, CAD Software, Blueprint Reading, Industrial Design, Mechanical Drawing, 2D/3D Modeling, Manufacturing Processes, Geometric Dimensioning, Technical Documentation, Drafting Standards",
            "Information Systems Management, Database Management, Systems Analysis, IT Strategy, Business Intelligence, Project Management, Data Analytics, Network Administration, Enterprise Systems, IT Governance",
            "HR Management, Talent Acquisition, Employee Relations, Compensation & Benefits, Training & Development, Organizational Development, Labor Laws, Performance Management, HR Strategy, Workforce Planning",
            "Electronic Testing, Quality Control, Test Procedures, Circuit Testing, Equipment Calibration, Technical Documentation, Troubleshooting, Quality Standards, Test Equipment Operation, Defect Analysis",
            "Aircraft Systems, Radar Technology, Communications Systems, Avionics Maintenance, Electronic Systems, Troubleshooting, Technical Documentation, Equipment Maintenance, System Integration, Safety Protocols",
            "Physical Principles, Mathematical Analysis, Laboratory Techniques, Data Analysis, Scientific Computing, Experimental Methods, Research Methodology, Problem Solving, Theoretical Physics, Scientific Writing",
            "GIS Software, Remote Sensing Analysis, Spatial Data Analysis, Map Creation, Environmental Monitoring, Data Collection, Image Processing, Geospatial Analysis, Environmental Assessment, Technical Report Writing",
            "Digital Marketing Strategy, Regional Marketing, E-commerce Management, Marketing Analytics, Campaign Management, Content Strategy, SEO/SEM, Market Analysis, Team Leadership, Budget Management",
            "Policy Analysis, Research Methods, Statistical Analysis, Public Administration, Economic Analysis, Policy Evaluation, Social Research, Government Relations, Data Analysis, Policy Development",
            "Basic French Language, Business French, Cultural Understanding, Oral Communication, Written Communication, Language Learning Methods, Vocabulary Building, Grammar Fundamentals, Pronunciation, Basic Business Communication",
            "Theological Research, Religious Studies, Pastoral Leadership, Ministerial Skills, Academic Writing, Religious Education, Counseling, Public Speaking, Theological Analysis, Community Leadership",
            "Mail Processing, Customer Service, Sorting Systems, Postal Regulations, Cash Handling, Package Handling, Address Verification, Mail Classification, Data Entry, Record Keeping",
            "Business Fundamentals, Management Principles, Accounting Basics, Marketing Fundamentals, Business Communication, Financial Management, Business Ethics, Team Management, Business Operations, Strategic Planning",
            "Danish Language, Danish Literature, Cultural Understanding, Written Communication, Oral Communication, Grammar Skills, Language Comprehension, Cultural Studies, Reading Comprehension, Communication Skills",
            "Software Development, Programming Languages, Database Management, System Analysis, Web Development, Application Design, Software Engineering, Network Programming, Operating Systems, Project Management",
            "Corporate Governance, Strategic Leadership, Executive Decision Making, Business Strategy, Change Management, Financial Management, Risk Management, Corporate Innovation, Global Business, Executive Communication",
            "Process Engineering, Chemical Process Design, Transport Phenomena, Reaction Engineering, Process Control, Advanced Mathematics, Research Methods, Laboratory Techniques, Process Optimization, Chemical Systems Analysis",
        ]
        generations = [r.split(", ") for r in generations]
#         prompt = """Please answer the skills can be gotten from the following content: 

# [Job]: B.S. in Computer Science
# [Skills]: Programming Languages, Operating Systems, Development Environment: Java, Python, C++, C, JavaScript, SQL, Linux/Unix, Windows, Git, VS Code, Eclipse, IntelliJ IDEA, Command Line Interface

# [Job]: {}"""
#         generations = self.llmeval.inference(
#             prompt=prompt,
#             data=[{'inputs':[r['initial_node']]} for r in queries],
#             max_tokens=256,
#             temperature=0.7,
#             batch_size=len(queries),
#             do_log=True
#         )
        return [{**r, 'requirements': g} for r, g in zip(queries, generations)]
    
    
    def retrieve(self, queries: List[dict], top_k: int, stage: int) -> List[List[str]]:
        new_queries = []
        if self.ret_type == 'lexical':
            top_k_results = em_multi_multi(
                queries=[r['requirements'] for r in queries],
                documents=self.documents,
                top_k=top_k
            )
        
        elif self.ret_type == 'semantic':
            top_k_results = self.retriever.retrieve_multi_multi(
                method='semantic',
                queries=[r['requirements'] for r in queries],
                documents=self.documents,
                top_k=top_k
            )
        for query, retrieved in zip(queries, top_k_results):
            for ret in retrieved:
                new_node = {
                    'content' : self.data[ret[0]]['job'],
                    'stage' : stage,
                    'parent_id' : query['parent_id'] + [query['node_id']],
                    'type' : 'ret',
                    'requirements' : list(set(query['requirements'] + ret[2])),
                    'node_id' : self._get_new_node_id(tree_id=query['graph_id']),
                    'graph_id': query['graph_id']
                    }
                self.tree_update(new_node)
                new_queries.append(new_node)
        return new_queries
    
    
    def _most_diverse(self, nodes: List[Dict], beam_size: int) -> List[Dict]:
        if beam_size in [None, 0]:
            return nodes
        
        if len(nodes) <= beam_size:
            return nodes
        
        contents = [r['content'] for r in nodes]
        embeddings = self.sentence_model.encode(contents)
        distances = cdist(embeddings, embeddings, 'cosine')
        
        # Maximum Marginal Relevance (MMR) 방식으로 구현
        selected_indices = []
        candidate_indices = list(range(len(nodes)))
        
        # 첫 번째 샘플은 random하게 선택
        first_idx = random.choice(candidate_indices)
        selected_indices.append(first_idx)
        candidate_indices.remove(first_idx)
        
        # 나머지 샘플들은 MMR로 선택
        while len(selected_indices) < beam_size:
            if not candidate_indices:
                break
                
            # 각 후보와 이미 선택된 샘플들 간의 최소 거리 계산
            candidate_scores = []
            for idx in candidate_indices:
                # 이미 선택된 샘플들과의 거리 중 최소값
                min_distance = min(distances[idx][j] for j in selected_indices)
                candidate_scores.append((idx, min_distance))
            
            # 가장 큰 최소 거리를 가진 샘플 선택
            next_idx = max(candidate_scores, key=lambda x: x[1])[0]
            selected_indices.append(next_idx)
            candidate_indices.remove(next_idx)
        
        return [nodes[i] for i in selected_indices]


    def beam_filtering(self, tree_searches: Dict[int, List[Dict]], min_distance_threshold: float = 0.1) -> Dict[int, List[Dict]]:
        filtered_searches = {}
        
        for graph_id, nodes in tree_searches.items():
            # 기본적인 다양성 필터링
            chosen_nodes = self._most_diverse(nodes, self.beam_size)
            
            # 추가적인 다양성 체크
            contents = [n['content'] for n in chosen_nodes]
            if self.beam_size and len(contents) > 1:
                embeddings = self.sentence_model.encode(contents)
                distances = cdist(embeddings, embeddings, 'cosine')
                
                # 대각선 제외한 최소 거리 계산
                min_distances = []
                for i in range(len(distances)):
                    row_distances = list(distances[i])
                    row_distances[i] = float('inf')  # 자기 자신과의 거리 제외
                    min_distances.append(min(row_distances))
                
                # 최소 거리가 threshold보다 작은 샘플들은 제외
                filtered_nodes = [node for node, min_dist in zip(chosen_nodes, min_distances) 
                                if min_dist >= min_distance_threshold]
                
                filtered_searches[graph_id] = filtered_nodes
            else:
                filtered_searches[graph_id] = chosen_nodes
        
        print(f"Filtered {len(tree_searches)} to {len(filtered_searches)}")

        try:
            del embeddings
            del distances
        except:
            pass
        gc.collect()              
        
        return filtered_searches


    def generate(self, queries: List[dict], top_k: int, stage: int) -> List[List[str]]:
        new_queries = []
        input_data = [{'inputs':[self._get_prompt(query, job_n=top_k, bias=self.do_bias)]} for query in tqdm(queries)]
        print(input_data[0]['inputs'][0])
        
        generations = self.llmeval.inference(
            prompt='{}',
            data=input_data,
            max_tokens=4096,
            temperature=self.temperature,
            batch_size=10,
            do_log=True,
            save_path='tmp.jsonl'
        )
        
        tree_searches = defaultdict(list)
        for llm_answer, query in zip(generations, queries):
            generations = self.parsing_function(llm_answer, top_k)
            for i, gen in enumerate(generations): # dict
                tree_searches[query['graph_id']].append({
                    'content' : gen['position'],
                    'stage' : stage,
                    'parent_id' : query['parent_id'] + [query['node_id']],
                    'type' : 'gen',
                    'node_id' : self._get_new_node_id(tree_id=query['graph_id']) + i,
                    'salary' : gen['salary'],
                    'year' : gen['year'],
                    'requirements' : gen['requirements'],
                    'graph_id': query['graph_id']
                })
                
        tree_searches = self.beam_filtering(tree_searches)
        for graph_id, queries in tree_searches.items():
            for query in queries:
                self.tree_update(query)
                new_queries.append(query)
                
        del input_data
        del queries
        return new_queries
    
    
    def _make_save_part(self, last_state: int = 0):
        if last_state == 0:
            return '_'.join([str(top_k)+method for top_k, method in zip(self.top_k_list, self.methods)])
        return '_'.join([str(top_k)+method for top_k, method in zip(self.top_k_list[:-1*last_state], self.methods[:-1*last_state])])
    
                
    def run(self, queries: List[dict], load_where: str) -> List[List[str]]:
        if self.methods[0] == 'ret' or self.methods[0] == 'retrieve':
            queries = self.annotate_skills(queries)
            self.trees = [{'initial_node': query['initial_node'],
                             'nodes': [{'content' : query['initial_node'],
                             'stage' : 0,
                             'node_id' : 0,
                             'requirements' : query['requirements'], # this is query of retrieval
                             'parent_id' : [],
                             'type' : 'question',
                             'graph_id' : query['graph_id']}],
                             'graph_id' : query['graph_id']} for query in queries]
        else:
            self.trees = [{'initial_node': query['initial_node'],
                             'nodes': [{'content' : query['initial_node'],
                             'stage' : 0,
                             'node_id' : 0,
                             'parent_id' : [],
                             'type' : 'question',
                             'graph_id' : query['graph_id']}],
                             'graph_id' : query['graph_id']} for query in queries]
        
        if self.retrieve_lookback:
            save_part = self._make_save_part(last_state=0)
            last_completed_stage = 0
            
            for stage in range(1, len(self.methods) + 1):
                stage_path = self.save_dir.replace(save_part, self._make_save_part(last_state=stage))
                if os.path.exists(stage_path):
                    self.trees = open_json(os.path.join(stage_path, f"{self.model_name_or_path.replace('/', '_')}.jsonl"))
                    last_completed_stage = len(self.methods) - stage
                    print(f"Found completed stage {stage}")
                    break

            if last_completed_stage > 0:
                print(f"Successfully loaded trees from stage {last_completed_stage}")
        else:
            last_completed_stage = 0
        # only leaf nodes
        queries = []
        for tree in self.trees:
            max_parent_length = max(len(node['parent_id']) for node in tree['nodes'])
            leaf_nodes = [node for node in tree['nodes'] if len(node['parent_id']) == max_parent_length]
            queries.extend(leaf_nodes)

        
        # Process each stage
        for i, (top_k, method) in enumerate(zip(self.top_k_list, self.methods)):
            current_stage = i + 1
            
            # Skip if this stage was already completed
            if current_stage <= last_completed_stage:
                print(f"Skipping completed stage {i} {top_k} {method}")
                continue

            if method == 'load':
                queries = self.load(queries, top_k, stage=current_stage, load_where=load_where)
            elif method in ['retrieve', 'ret']:
                # Allow retrieval from multiple stages back
                retrieve_stage = max(0, current_stage - self.retrieve_lookback) if hasattr(self, 'retrieve_lookback') else current_stage - 1
                queries = self.retrieve(queries, top_k, stage=current_stage, from_stage=retrieve_stage)
            elif method in ['generate', 'gen']:
                queries = self.generate(queries, top_k, stage=current_stage)
                gc.collect()
                torch.cuda.empty_cache()
            elif method == 'pivot_gen':
                queries = self.pivot_gen(queries, top_k, stage=current_stage)
            else:
                raise ValueError(f"Invalid method: {method}")
                
            print(f"Stage {current_stage}, {method}, {top_k}: {len(queries)} nodes generated")
            if self.save_tmp:
                os.makedirs(os.path.join(self.save_dir, f'tmp_{i}'), exist_ok=True)
                save_json(self.trees, os.path.join(self.save_dir, f'tmp_{i}', f"{self.model_name_or_path.replace('/', '_')}.jsonl"))
        
        return self.trees

    