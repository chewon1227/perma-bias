import json
import fire
import os
import glob

def find_json_files(directory):
    directory = os.path.abspath(directory)    
    json_files = []

    for root, dirs, files in os.walk(directory):
        for file in glob.glob(os.path.join(root, '*.json')):
            json_files.append(os.path.abspath(file))
    
    return json_files


def inspect(
    path: str = '/scratch2/sb/career-ytb-scripts-en_0-0'
):
    done = 0
    undone = 0
    result = []
    files = find_json_files(path)
    for file in files:
        with open(file) as f:
            data = json.load(f)
            try:
                if len(data["yt_meta_dict"]['subtitles']) == 4:
                    done += 1
                    result.append(data['url'])
                    continue
            except:
                undone += 1
                
            try:
                if len(data["clip_subtitles"][0]['lines']) == 4:
                    done += 1
                    result.append(data['url'])
            except:
                undone += 1
                
    print(f"Done: {done}, Undone: {undone}")
    return list(set(result))
    
if __name__ == '__main__':
    fire.Fire(inspect)