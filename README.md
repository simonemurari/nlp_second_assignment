
# Second assignment

## Slicing of excessive context window for LLMs

Large Language Models can be trained for a number of purposes. In this assignment, candidates are either required to (CASE 1) implement an algorithm to generate slicing of excessive context window for ChatGPT 3.5, or (CASE 2) to implement a hierarchical system for summarization in the same system. A correct submission will just consist in one of the two cases. Please do not submit gits for both cases. The implementation for both choices are to be either in Python, while using NLTK, or in Java while using OpenNLP.

CASE 1

The method is based on the following pipeline:

* When the input is below the standard size of the context window (128 Mb) is then passed "as it is" to the LLM;

* When the input is above the standard size is subdivided in a finite number of slices each of a size that
   fits the context window and such that they sum to a number N greater than or equal to the size of the input
   length;

* The criteria to generate a coverage as provided above are:

      ** Two slices can overlap;
      ** No slice is included in another one;
      ** When two adjacent slices are settled, the two slices have to be different "enough".

Ideal solutions will be based on the comparison of two slices based on cosine distance of bag of words constructed by the usual pipeline of stopword elimination, stemming/lemmatization and count of occurrences weighted on the length of the document after the steps above. The setup of the threshold for distance is empirical, no need to settle it by experiments (use reasonable threshold like 20%).

Once the prompt engineering algorithm has been run, we shall collect the results and use them as they are, so the assignment does not require ex-post filtering.

## Comments

I implemented this assignment in Python, using the libraries NLTK, sklearn and llama-cpp-python. The code is contained in the notebook `second_nlp_assignment.ipynb` which is commented and already run so you can see the results without downloading it and running it again.\
\
Instead of using ChatGPT 3.5 I used the model `mistral-7b-instruct-v0.2.Q4_K_M` (that I downloaded from [here](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)) from Mistral AI thanks to the library llama_cpp and initialized it with a context window of 4096 tokens.
The input files are 10 pages from Wikipedia and they are inside the folder `pages`.\
For every page I did a light preprocessing: I tokenized the text with the word tokenizer and removed the stopwords using NLTK and then with a regex I fixed some punctuation errors.\
After that I tokenized the text with the tokenizer of the model and I started generating the slices. To generate the slices I used the function `generate_slices` that takes in input the text, the maximum length of the context window, the maximum overlap between slices and the threshold for the cosine distance. To pass the text to the cosine distance function I had to detokenize the text, pass it to the Count Vectorizer to calculate the BoW matrix and then calculate the cosine distance between the previous slice and the current one. In the end I returned the generated slices.\
\
Finally, after generating the slices, I passed each slice to the model asking it to give me the main topic of each slice and then I printed the results.\
In the notebook you can have a more in-depth look at the commented code and the results.