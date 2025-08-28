from mylmeval import get_results, open_json
import fire
from infer_llm_kaggle import process_data, INPUT_FORMATS

prompt = """Translate the following job description to {}:

[Original]:
1. {}
2. {}
[Translated]:"""

languages = ['es', 'en', 'ja', 'ko']
FULL_LANGUAGE = {
    'es' : 'Español',
    'en' : 'English',
    'ja' : '日本語',
    'ko' : '한국어',
    }


def infer_llm():
    p_data = []
    for lang in languages:
        t_langs = [r for r in languages if r != lang]
        for t_lang in t_langs:
            p_data += [{
                'inputs' : [FULL_LANGUAGE[t_lang], r['groundtruth'], *r['inputs']],
                'groundtruth' : r['groundtruth'],
                'source_lang' : lang,
                'translated_lang' : t_lang,
                } for r in process_data(lang, lang, INPUT_FORMATS[lang])]
    import pdb; pdb.set_trace()
    print(len(p_data))
    _ = get_results(
            model_name_or_path='gpt-4o-mini',
            prompt=prompt,
            data=p_data,
            do_log=True,
            batch_size=1,
            max_tokens=1024,
            apply_chat_template=False,
            save_path=f"data/translated_kaggle.jsonl"
            )
    print(f"Results saved in data/translated_kaggle.jsonl") 


if __name__ == '__main__':
    fire.Fire(infer_llm)
