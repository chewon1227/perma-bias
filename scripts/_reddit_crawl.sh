#!/bin/bash
KEYWORDS=("jobs" "CareerChange" "findapath" "jobhunting" "Employment" "ITCareerQuestions")
KEYWORDS=("jobsearchhacks" "WorkOnline" "hiring" "jobpostings" "GetEmployed" "jobsearch" "Resumes" "recruitinghell" "Managers")
KEYWORDS=("Work" "WorkReform" "GetMotivated" "Productivity" "CareerQuestions" "burnedout" "remotework" "AskHumanResour" "recruiting")


# Loop through each keyword
for keyword in "${KEYWORDS[@]}"; do
    echo "Processing keyword: $keyword"
    
    # Run the parallel scraping for each keyword
    # python datacuration/2_reddit.py batch \
    #     --keyword="$keyword" \
    #     --sort_type="new" \
    #     --start=0 \
    #     --batch_size=50

    python datacuration/2_reddit.py parallel \
        --keyword=$keyword \
        --sort_type="new" \
        --start=0 \
        --max_workers=3
    
    # Optional: Add a delay between keywords to avoid rate limiting
    sleep 5
    
    echo "Completed processing for: $keyword"
    echo "-----------------------------------"
done

echo "All keywords processed successfully!"