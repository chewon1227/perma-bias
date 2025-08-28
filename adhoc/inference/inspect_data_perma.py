from careerpathway.llm import get_results
from careerpathway.data import load_reddit_data
from careerpathway.utils import open_json, save_json
import os
import matplotlib.pyplot as plt
from termcolor import colored
import random
import itertools
import numpy as np

def _analyse_perma(results, colors, data, prompt_version):
    """Aggregate and visualize PERMA results.

    Parameters
    ----------
    results : list[dict]
        Each dict contains at least a "result" and the original "prompt" (added by `get_results`).
    colors : list[str]
        Color palette.
    data : list[dict]
        Original data items (title, content) used to build the prompt – needed for examples.
    prompt_version : str
        Either 'v1' or 'v2', determines how to parse the model output.
    """
    print(f"Analyzing {len(results)} results with prompt version {prompt_version}")
    categories = ['Positive Emotion', 'Engagement', 'Relationships', 'Meaning', 'Accomplishment']
    # store titles associated with each category for later sampling/printing
    examples_map: dict[str, list[str]] = {cat: [] for cat in categories}

    def _update_examples(cat: str, title: str):
        if len(examples_map[cat]) < 30:  # keep at most 30 titles per cat to limit memory
            examples_map[cat].append(title)

    # ---------------------------------------------------------------------
    # Parse each result depending on prompt_version
    # ---------------------------------------------------------------------
    for r, d in zip(results, data):
        title = d['inputs'][0]
        res_raw = r.get('result', '')
        res_lower = res_raw.lower()

        if prompt_version == 'v1':
            # v1 returns comma-separated PERMA domains; simple containment check
            for cat in categories:
                if cat.lower() in res_lower:
                    _update_examples(cat, title)

        elif prompt_version == 'v2':
            # Expected two lines: Options: ..., PERMA: ...
            perma_line = ''
            for line in res_raw.splitlines():
                if line.lower().startswith('perma:'):
                    perma_line = line[len('PERMA:'):].strip()
                    break
            # perma_line now like "overall-meaning" or "job a-meaning, job b-relationships"
            for token in perma_line.split(','):
                token = token.strip()
                if '-' in token:
                    _, domain = token.rsplit('-', 1)
                else:
                    domain = token
                domain = domain.strip().title()
                if domain in categories:
                    _update_examples(domain, title)

            # Build set of domains for this post for later co-occurrence
            domains_in_post = set()
            for token in perma_line.split(','):
                token = token.strip()
                if '-' in token:
                    _, domain = token.rsplit('-', 1)
                else:
                    domain = token
                domain = domain.strip().title()
                if domain in categories:
                    domains_in_post.add(domain)

            # Track none vs pairs
            if not domains_in_post:
                none_count = locals().get('none_count', 0)
                none_count += 1
                locals()['none_count'] = none_count
            else:
                # combinations of size 2
                pair_matrix = locals().get('pair_matrix')
                if pair_matrix is None:
                    pair_matrix = np.zeros((len(categories), len(categories)), dtype=int)
                    locals()['pair_matrix'] = pair_matrix
                for a, b in itertools.combinations(domains_in_post, 2):
                    i, j = categories.index(a), categories.index(b)
                    pair_matrix[i, j] += 1
                    pair_matrix[j, i] += 1  # symmetric

        else:
            raise ValueError(f"Unsupported prompt_version {prompt_version}")

    # ---------------------------------------------------------------------
    # Prepare counts & sample examples
    # ---------------------------------------------------------------------
    counts = {cat: len(titles) for cat, titles in examples_map.items()}

    for cat in categories:
        print(colored(f"{cat}: {counts[cat]}", 'green'))
        sample_titles = random.sample(examples_map[cat], min(len(examples_map[cat]), 10))
        for idx, t in enumerate(sample_titles, 1):
            print(colored(f"Example {idx}:", 'red'), t)
        print('-' * 30)

    # ---------------------------------------------------------------------
    # Bar plot
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(categories)), counts.values(), color=colors[1])

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height / 2 if height > 0 else 0,
                f"{int(height)}", ha='center', va='center', color='white', fontsize=12)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=30, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of PERMA Difficulties')
    plt.tight_layout()
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/reddit_data_perma_barplot.png', bbox_inches='tight')

    # ---------------------------------------------------------------------
    # Heatmap for v2 co-occurrence
    # ---------------------------------------------------------------------
    if prompt_version == 'v2':
        pair_matrix = locals().get('pair_matrix', np.zeros((len(categories), len(categories)), dtype=int))
        none_count = locals().get('none_count', 0)

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        im = ax2.imshow(pair_matrix, cmap='Blues', vmin=0)

        # ticks and labels
        ax2.set_xticks(range(len(categories)))
        ax2.set_yticks(range(len(categories)))
        ax2.set_xticklabels(categories, rotation=45, ha='right')
        ax2.set_yticklabels(categories)

        # annotate
        for i in range(len(categories)):
            for j in range(len(categories)):
                if i == j:
                    continue
                val = pair_matrix[i, j]
                if val > 0:
                    ax2.text(j, i, str(val), ha='center', va='center', color='white' if val > pair_matrix.max()/2 else 'black')

        ax2.set_title(f'PERMA Pairwise Co-occurrence (None: {none_count})')
        plt.tight_layout()
        plt.savefig('results/plots/reddit_data_perma_heatmap.png', bbox_inches='tight')


def main(
    model_name_or_path: str = "gpt-4o-mini",
    only_parsing: bool = False,
    prompt_version: str = 'v1',
    colors: list[str] = ['black', '#4BDBC5', 'lightgray'],
    sample_n: int = 1000
):
    os.makedirs('results/tmp_data', exist_ok=True)
    data = load_reddit_data(sampling_num=sample_n if 'gpt' in model_name_or_path or 'o1' in model_name_or_path else None)

    if only_parsing:
        file_path = f"results/tmp_data/reddit_data_perma_{model_name_or_path.replace('/', '_')}_{sample_n}_{prompt_version}.jsonl"
        results = open_json(file_path)
        _analyse_perma(results, colors, data, prompt_version)
        return


    if prompt_version == 'v1':
        prompt = """
You are a psychologist applying the PERMA model. Read the career-related Reddit post below and identify which PERMA domains the writer is struggling with. Multiple choices are possible, but choose only those that clearly stand out.

- Positive Emotion: hope, joy, interest, compassion, pride, amusement
- Engagement: flow, absorption
- Relationships: social ties, family, friends, bosses, community
- Meaning: purpose, growth, contribution
- Accomplishment: achievement, mastery, competence

Return the selected domains separated by commas, **with no extra text**.

[Title]: {0}
[Content]: {1}
"""
    elif prompt_version == 'v2':
        prompt = """
You are a psychologist applying the PERMA model.

Task 1 – Options:
Determine whether the writer is explicitly weighing several concrete options (e.g., two job offers, study vs work). If yes, list the options separated by semicolons. If no clear alternatives are mentioned, write "None".

Task 2 – Salient PERMA value for each option:
For every option you listed (or for the single situation if "None"), choose ONE PERMA domain that stands out the most:
Positive Emotion, Engagement, Relationships, Meaning, Accomplishment.

Output format (exactly):
Options: <None | option1; option2; ...>
PERMA: <option1>-<Domain>, <option2>-<Domain>, ...>  # If Options is None, use Overall-<Domain>

Return NOTHING else.

[Title]: {0}
[Content]: {1}
"""

    results = get_results(
        model_name_or_path=model_name_or_path,
        prompt=prompt,
        data=data,
        batch_size=1,
        max_tokens=2000,
        save_path=f'results/tmp_data/reddit_data_perma_{model_name_or_path.replace("/", "_")}_{len(data)}_{prompt_version}.jsonl'
    )

    _analyse_perma(results, colors, data, prompt_version)
    return results


if __name__ == "__main__":
    import fire
    fire.Fire(main)

