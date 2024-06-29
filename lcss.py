
import json

ntasnames = json.load(open("ntanames.json"))
pdfile = json.load(open("popdensity.json"))

def longest_common_substring(str1, str2):
    m = len(str1)
    n = len(str2)
    # Create a table to store lengths of longest common suffixes of substrings
    # Initialize the table with 0
    table = [[0] * (n + 1) for _ in range(m + 1)]
    # Variable to store the length of the longest common substring
    longest_length = 0
    # Variable to store the ending index of the longest common substring
    ending_index = 0
    # Fill the table and update the longest_length and ending_index accordingly
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
                if table[i][j] > longest_length:
                    longest_length = table[i][j]
                    ending_index = i
            else:
                table[i][j] = 0
    # Return the length of the longest common substring
    return longest_length

def get_top_n_strings(string, string_list, n):
    # Calculate the longest common substring between the input string and each string in the list
    common_substrings = [(s, longest_common_substring(string, s)) for s in string_list]
    # Sort the list of common substrings in descending order based on the length
    sorted_substrings = sorted(common_substrings, key=lambda x: x[1], reverse=True)
    # Get the top n strings with the longest common substrings
    top_n_strings = [s[0] for s in sorted_substrings[:n]]
    return top_n_strings

with open("namepossibilities.json", "w") as np:
    output = []
    for nta in pdfile:
        output.append({"origname": nta["ntaname"], "possibilities": get_top_n_strings(nta["ntaname"], ntasnames, 5), "index": 0})
    json.dump(output, np)

