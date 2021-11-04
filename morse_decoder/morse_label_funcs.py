import numpy as np

code_number = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
]

letter_to_morse_dict = {
        "a": [1,0,1,1,1],
        "b": [1,1,1,0,1,0,1,0,1],
        "c": [1,1,1,0,1,0,1,1,1,0,1],
        "d": [1,1,1,0,1,0,1],
        "e": [1],
        "f": [1,0,1,0,1,1,1,0,1],
        "g": [1,1,1,0,1,1,1,0,1],
        "h": [1,0,1,0,1,0,1],
        "i": [1,0,1],
        "j": [1,0,1,1,1,0,1,1,1,0,1,1,1],
        "k": [1,1,1,0,1,0,1,1,1],
        "l": [1,0,1,1,1,0,1,0,1],
        "m": [1,1,1,0,1,1,1],
        "n": [1,1,1,0,1],
        "o": [1,1,1,0,1,1,1,0,1,1,1],
        "p": [1,0,1,1,1,0,1,1,1,0,1],
        "q": [1,1,1,0,1,1,1,0,1,0,1,1,1],
        "r": [1,0,1,1,1,0,1],
        "s": [1,0,1,0,1],
        "t": [1,1,1],
        "u": [1,0,1,0,1,1,1],
        "v": [1,0,1,0,1,0,1,1,1],
        "w": [1,0,1,1,1,0,1,1,1],
        "x": [1,1,1,0,1,0,1,0,1,1,1],
        "y": [1,1,1,0,1,0,1,1,1,0,1,1,1],
        "z": [1,1,1,0,1,1,1,0,1,0,1]
    }

letter_to_morse_dict_categorical = {
        "a": [1,2],
        "b": [2,1,1,1],
        "c": [2,1,2,1],
        "d": [2,1,1],
        "e": [1],
        "f": [1,1,2,1],
        "g": [2,2,1],
        "h": [1,1,1,1],
        "i": [1,1],
        "j": [1,2,2,2],
        "k": [2,1,2],
        "l": [1,2,1,1],
        "m": [2,2],
        "n": [2,1],
        "o": [2,2,2],
        "p": [1,2,2,1],
        "q": [2,2,1,2],
        "r": [1,2,1],
        "s": [1,1,1],
        "t": [2],
        "u": [1,1,2],
        "v": [1,1,1,2],
        "w": [1,2,2],
        "x": [2,1,1,2],
        "y": [2,1,2,2],
        "z": [2,2,1,1]
    }



def labels_to_one_hot_positions_categorical(morse_words, letter_n):

    morse_words = morse_words[0:,:1].reshape(-1,)

    morse_labels_one_hot = []

    for morse_word in morse_words:

        # use npzeros(letter_n) here !!
        temp_arr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        for c in range(len(morse_word)):
            letter = morse_word[c]
            temp_arr[c] = code_number.index(letter) +1
            
        n = np.atleast_2d(temp_arr).T
        morse_labels_one_hot.append(n)

    return np.array(morse_labels_one_hot)    


def letter_n_to_one_hot_positions_categorical(morse_words, letter_n):

    morse_words = morse_words[0:,:1].reshape(-1,)
    
    morse_labels_one_hot = []

    for morse_word in morse_words:
        
        if (letter_n + 1 > len(morse_word)):
            n = np.atleast_2d(0).T
        else:
            n = np.atleast_2d(code_number.index(morse_word[letter_n]) +1).T

       
        morse_labels_one_hot.append(n)

    return np.array(morse_labels_one_hot)  


def position_regression(morse_words, letter_n, image_target_size):

    mw = (morse_words[0:, letter_n: letter_n +1].reshape(-1,))

    print(image_target_size)
    
    return  mw.astype(np.float) / image_target_size


    
def labels_to_one_hot(morse_words):

    morse_labels_one_hot = []

    for i in range(len(morse_words)):
        temp_arr = np.zeros(26)
        for c in range(len(morse_words[i])):
            letter = morse_words[i][c]
            temp_arr[code_number.index(letter)] = 1

        morse_labels_one_hot.append(np.array(temp_arr))

    return np.array(morse_labels_one_hot)