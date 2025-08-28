from careerpathway.utils import open_json
from careerpathway.data import inspect_data
from termcolor import colored

category_1_job_search_career_transition = [
    "Is it too late for a career change?",
    "How do careers work?",
    "Major career change at 50",
    "People who transitioned careers but it didn't work out… what did you do after?",
    "In college and don't know what the heck to do with your life? Start by working backwards",
    "What do you think of accepting a job and still hunting for another one ?",
    "Should you re-apply to a job that has turned you down already?",
    "Help! I think it's time to move on.",
    "Gf is stuck in a rut",
    "I want to change job because of toxic manager. Will it be seen as a red flag?",
    "How does everyone keep a positive mindset when applying for jobs in this market?",
    "Is it really that impossible to land a full-time role?",
    "College or $70K Job?",
    "Jobs for people afraid of everything?",
    "I'm willing to do soul crushing work for long hours if it means getting paid extremely well. What careers should I consider?",
    "Will the remote/hybrid trend stay??",
    "I've come to learn that getting a job is about who you know",
    "Jobs where travel is borderline mandatory and you're gone for extended periods of time.",
    "Should I turn in resignation on Monday or Friday?",
    "What are some jobs that are overlooked and are currently hiring",
    "How do people actually succeed and make $120k + a year?",
    "Online jobs to do while traveling",
    "Turning 28 tomorrow, is it too late to turn my life around?",
    "How difficult is managing Gen Z Employees",
    "Is Higher Education Still Worth It in Today's Job Market?",
    "What's the story that you'd want a hiring team to know that's not on your resume?",
    "How do recent grads get jobs not related to their degree? (Please don't just say networking!)",
    "Is the job market dead?",
    "How do you guys search for jobs?",
    "Major career change at 50",
    "What did you do after high school",
    "I was a 39yr old that 'learned to code' 5 years ago, now I am doing well as a data engineer.",
    "We're toast, the job market is broken maybe beyond repair.",
    "Can we have a sticky at the top reminding people that they're probably not too old to do a career change?",
    "I'm a 30-year-old failure",
    "I finally got the job. It felt impossible but it finally happened!!!",
    "struggling to get back on track",
    "What is going on with hiring these days?",
    "How long has it been since you've been laid off & unemployed?",
    "3 years out of college no real job",
    "Is business administration a worthless degree?",
    "what to do as a stay at home wife?",
    "Dealing with job search anxiety",
    "I need work A.S.A.P.",
    "34M (Toronto, Ontario)- Failed to achieve anything in life, unemployed, no career, no relationship, not even one single friend. Absolute failure in all categories and in major panic mode",
    "I'm out of confidence to look for a job",
    "When do you think the job market for recruiters will pick up?",
    "It's not getting better",
    "How do you offer references without giving away that you're looking for a new job"
]

# Category 2: Life Direction & Purpose
category_2_life_direction_purpose = [
    "33M feeling lost in life",
    "I'm 20 and reached nowhere in life yet",
    "Feeling like I am here at this point",
    "i'm so fucking lost",
    "I feel like my job has no real value and this has been eating me alive",
    "What's your biggest regret in life? Let people learn from your regrets and mistakes in life.",
    "25F Loser, Unemployed, No Car, No Friends, No Family, No Significant Other (USA)",
    "Am I always going to feel like I need to learn everything?",
    "\"Felt Like My 20s Just Began, But 30s Are Already Around the Corner\"",
    "I'm becoming irrelevant. What next?",
    "is it possible to fix your life in your 30s?",
    "I don't think I'm taking life seriously enough",
    "Is it possible to do more than just survive?",
    "Any advice for my friend who's lost hope about ever amounting in life?",
    "Thinking About Dropping Out Right Before Finishing My Master's—Will I Regret This Forever?",
    "I was desperate for a job and now I regret. What should I do?",
    "How do I stop feeling guilt about looking for a new job?",
    "Is it worth being miserable for a good job?",
    "Tomorrow is a lie.",
    "What's one thing you're deeply passionate about?",
    "If you were not in need of a job, how would you structure your day?",
    "How do you find the strength to keep going even though your best is not enough?",
    "Turning 28 tomorrow, is it too late to turn my life around?",
    "At what point does determination become delusion?",
    "I'm a 30-year-old failure",
    "What do you do when you only have savings to cover you for two months and you need a job now?",
    "Not sure what to do",
    "Should I Be Worried?",
    "How to motivate yourself?",
    "I'm out of confidence to look for a job",
    "Has anyone beat the constant anxious and pessimistic thinking?",
    "It's not getting better",
    "What's your biggest problem in leadership",
    "Maybe I am not cut out for this?",
    "I need to easily make 6 figures in life. I love to problem solve and always need to be doing something. What are my options?",
    "I can't quit my job but it's killing me…",
    "Wisdom of Age"
]


def check_reddit_reply(
    data_path: str = 'data/all_with_comments.jsonl',
    print_title: bool = True,
    print_post: bool = True,
    print_comments: bool = False,
):
    
    cnt = 0
    subsample = inspect_data(data_path, columns=['Title', 'Post Text', 'comments'], random_n=10000, do_print=False)
    for item in subsample:
        comment_scores = [comment['score'] for comment in item['comments'] if comment['score'] > 10]
        if len(comment_scores) > 0 and (item['Title'] in category_1_job_search_career_transition or item['Title'] in category_2_life_direction_purpose):
            if print_title:
                print(colored('Title', 'green'), item['Title'])
            if print_post:
                print(colored('Post Text', 'green'), item['Post Text'])
            if print_comments:
                print(colored('comments', 'green'))
                for comment in item['comments']:
                    print(colored('comment', 'red'), comment['body'], colored(f"({comment['score']})", 'magenta'))
            print('-'*100)
            cnt += 1
    print(f"Total: {cnt}")
if __name__ == "__main__":
    import fire
    fire.Fire(check_reddit_reply)