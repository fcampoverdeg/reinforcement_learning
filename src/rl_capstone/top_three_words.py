'''
Create a program that tracks and displays the top three most frequent words in a given list.

Example:
Input: ["pp", "hard", "hi", "subscribe", "pp", "hard", "subscribe", "pp"]
Output: ["pp", "hard", "subscribe"]
'''

from collections import Counter

# ------------------------------------------------------------
# Step 1: Initialize variables
# ------------------------------------------------------------
default_words = ["pp", "hard", "hi", "subscribe", "pp", "hard", "subscribe", "pp"]
result = []


# ------------------------------------------------------------
# Step 2: Ask the user for a list of words (or use the default)
# ------------------------------------------------------------
def input_from_user(default_words):

    while True:
        
        # Take user input
        user_input = input("Enter the number of words for the list. (Press Enter to use default list.): ")

        # If user presses Enter, use default list of words
        if user_input.strip() == "":
            print("Using the default list of words: ", default_words)
            #  my_dic = default_words.copy()   We cannot do this, because it will copy exactly a list into the dictinonary
            words = default_words
            break

        try:
            number_words = int(user_input)
            words = []

            for i in range(number_words):
                word = input("Enter a word: ")
                words.append(word)

            break   # exit the loop after valid input
                
        except ValueError:
            # If user input is not an integer
            print(f"Enter a valid integer or press Enter for default list.\n")

        
    return words

        
# Step 3: Fucntion to count the times a word repeats
def count_words(words):
    
    word_counts = {}         # Create an empty dictionary to store counts

    # Go through each word in the dictionary
    for word in words:
        # If that word exists in the dictionary, increase its counter
        if word in word_counts:
            word_counts[word] += 1
        # Otherwise, add it with a count of 1
        else:
            word_counts[word] = 1
        
    return word_counts


# ------------------------------------------------------------
# Step 4: Find the top three most frequent words
# ------------------------------------------------------------
def top_three_words(word_counts):
    
    top_three = []      # List to store the top 3 words
    
    # Convert dictionary into a list of tuples (words, count)
    items = list(word_counts.items())

    # Sort by count in descending order (largest first)
    items.sort(key=lambda pair: pair[1], reverse=True)

    # Take only the 3 first words
    for i in range(min(3, len(items))):
        top_three.append(items[i][0])

    return top_three

    
# ------------------------------------------------------------
# Step 5: Combine everything
# ------------------------------------------------------------
words_from_user = input_from_user(default_words)
counts = count_words(words_from_user)
result = top_three_words(counts)

print("\nThe top 3 most frequent words are: ", result)


print("\n-----------------------------------------------------------\n")

################################################################################################################

my_dic = {}

# looking through the dictionary
for word in default_words:
    my_dic[word] = 1 + my_dic.get(word, 0)

# Sort based on reverse Values

# Dictionary Comprehension - Builds a 'new dictionary' directly from the sorted list
#    What it does:
#           -> Iterates through each (k, v) pair in the sorted list.
#           -> Creates key-value pairs in a 'new dictionary'
#           -> Keeps the order of sorting
#
#   Type of result:
#           dict
#
#   output: {'pp': 3, 'hard': 2, 'subscribe': 2, 'hi': 1}

dictionary_comprehension = {k: v for k, v in sorted(my_dic.items(), key=lambda items: items[1], reverse=True)}

result = []

for i, (word, count) in enumerate(dictionary_comprehension.items()):
    if i == 3:
        break
    result.append(word)

print(result)


################################################################################################################

# Sorted List of Tuples - Returns a 'list' of key-value pairs sorted by the value
#   What it does:
#           -> Sorts all items from 'my_dic' by value (descending orer)
#           -> Produces a 'list', not a dictionary
#           -> You can still loop through it or rebuild a dictionary later if needed.
#
#   Type of result:
#           list (of tuples)
#
#   output: [('pp', 3), ('hard', 2), ('subscribe', 2), ('hi', 1)]

sorted_items = sorted(my_dic.items(), key=lambda items: items[1], reverse=True)

result = []

for i in range(min(3, len(sorted_items))):
    result.append(sorted_items[i][0])


print(result)

################################################################################################################



