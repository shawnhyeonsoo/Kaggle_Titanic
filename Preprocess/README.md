1. Filling in values for empty instances in Ages from titles such as 'Mr.'
- Average age for each title
2. Changing Fare values into binary numbers
3. (To-dos) Filling in ages from Title + Sibling + Parch etc.
</br>
</br>
Feb. 23, 2021: </br>

- Changed preprocessing of age and fare data: binary --> boundary values </br>
- Age Boundary: 9 groups of 10 values each (0~9, 10 ~ 19, 20~ 29, 30~ 39, 40 ~ 49 etc)
- Fare Boundary: 6 groups of 100 values each (0~99, 100~199, 200~299, 300~ 399 etc)
- Parch: one hot vector
- Sibs: one hot vector </br>
   e.g. Age: 42 --> 0000100000  (Belonging to age group 5) </br>
        Fare: 50 --> 1000000 (Belonging to fare group 0) </br>
- Filled in missing age values as average of the ages with the same title (Mr. etc), after getting rid of the max and min value.
- Obtained an increased accuracy of 0.77033. </br>
