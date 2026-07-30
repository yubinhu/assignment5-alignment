# Write up

## `prompting_baselines.py`

- (a) I don't see any that has an correct answer but incorrect format. Full results in `results/tasks.txt`.
    ```bash
    FCAC=244, FCAI=1291, FIAC=0, FIAI=2422
    Question only fully correct: 0
    Question only format correct: 0
    Question only answer correct: 0
    Question only incorrect: 1319
    R1 zero fully correct: 0
    R1 zero format correct: 241
    R1 zero answer correct: 0
    R1 zero incorrect: 1078
    R1 zero three shot GSM8K fully correct: 244
    R1 zero three shot GSM8K format correct: 1050
    R1 zero three shot GSM8K answer correct: 0
    R1 zero three shot GSM8K incorrect: 25
    ```

- (b) For "question only" specifically, the model also sometimes finishes right away with an EOF. Overall, sometimes the model tries to generate multi-turn conversations between the user and the assistant. Few shots help with the first problem. 

