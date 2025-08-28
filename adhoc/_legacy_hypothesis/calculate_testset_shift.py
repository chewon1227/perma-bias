
import os
import fire 
from mylmeval.llm import MyLLMEval
from mylmeval.utils import open_json
from careerpathway.scoring.load_testset import load_high_qual_diversity
from careerpathway.scoring.diversity import Diversity
from datetime import datetime
nowdate = datetime.now().strftime("%Y%m%d")


def testset_shift(
):
    diversity = Diversity()
    testset, _ = load_high_qual_diversity(test_size=250, do_keyword=True, only_main=True) # 'Psychology'
    
    # 1. global diversity
    testset = [[item['initial_node']]+item['nodes'] for item in testset]
    globalset = [n for item in testset for n in item]
    global_diversity = diversity.evaluate([globalset])
    
    # 2. decoding diversity
    localset = testset
    local_diversity = diversity.evaluate(localset)
    
    return
        

if __name__ == '__main__':
    fire.Fire(testset_shift)