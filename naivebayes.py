from __future__ import division
import sys
import os.path
import numpy as np
from collections import Counter
import util
import math

USAGE = "%s <test data folder> <spam folder> <ham folder>"


def get_counts(file_list):
    """
    Computes counts for each word that occurs in the files in file_list.

    Inputs
    ------
    file_list : a list of filenames, suitable for use with open() or
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the number of files the
    key occurred in.
    """
    ### TODO: Comment out the following line and write your code here
    # raise NotImplementedError
    word_count = Counter()
    for file in file_list:
        words = set(util.get_words_in_file(file))
        word_count.update(words)

    return word_count


def get_log_probabilities(file_list, alpha=1.0):
    """
    Computes log-probabilities for each word that occurs in the files in
    file_list.

    Input
    -----
    file_list : a list of filenames, suitable for use with open() or
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the log of the smoothed
    estimate of the fraction of files the key occurred in.

    Hint
    ----
    The data structure util.DefaultDict will be useful to you here, as will the
    get_counts() helper above.
    """
    ### TODO: Comment out the following line and write your code here
    # raise NotImplementedError
    file_num = len(file_list)
    word_count = get_counts(file_list)
    total_words = len(word_count)
    words_log = util.DefaultDict(
        lambda: math.log(alpha / (file_num + total_words * alpha))
    )
    for word in word_count:
        smoothed_fraction = (word_count[word] + alpha) / (
            file_num + total_words * alpha
        )
        words_log[word] = math.log(smoothed_fraction)

    return words_log


def learn_distributions(file_lists_by_category):
    """
    Input
    -----
    A two-element list. The first element is a list of spam files,
    and the second element is a list of ham (non-spam) files.

    Output
    ------
    (log_probabilities_by_category, log_prior)

    log_probabilities_by_category : A list whose first element is a smoothed
                                    estimate for log P(y=w_j|c=spam) (as a dict,
                                    just as in get_log_probabilities above), and
                                    whose second element is the same for c=ham.

    log_prior_by_category : A list of estimates for the log-probabilities for
                            each class:
                            [est. for log P(c=spam), est. for log P(c=ham)]
    """
    ### TODO: Comment out the following line and write your code here
    # raise NotImplementedError
    spam_files, ham_files = file_lists_by_category
    spam_log_probs = get_log_probabilities(spam_files)
    ham_log_probs = get_log_probabilities(ham_files)
    log_probabilities_by_category = [spam_log_probs, ham_log_probs]
    spam_num = len(spam_files)
    ham_num = len(ham_files)
    total_num = spam_num + ham_num
    spam_log_prior = math.log(spam_num / total_num)
    ham_log_prior = math.log(ham_num / total_num)
    log_priors_by_category = [spam_log_prior, ham_log_prior]
    return (log_probabilities_by_category, log_priors_by_category)


def classify_message(
    message_filename,
    log_probabilities_by_category,
    log_prior_by_category,
    names=["spam", "ham"],
):
    """
    Uses Naive Bayes classification to classify the message in the given file.

    Inputs
    ------
    message_filename : name of the file containing the message to be classified

    log_probabilities_by_category : See output of learn_distributions

    log_prior_by_category : See output of learn_distributions

    names : labels for each class (for this problem set, will always be just
            spam and ham).

    Output
    ------
    One of the labels in names.
    """
    ## TODO: Comment out the following line and write your code here
    spam_log_probs, ham_log_probs = log_probabilities_by_category
    spam_log_prior, ham_log_prior = log_prior_by_category
    words_in_message = set(util.get_words_in_file(message_filename))
    # Initialize total log probabilities for each category
    # s = 0.9999
    total_log_prob_spam = spam_log_prior
    total_log_prob_ham = ham_log_prior

    # Accumulate log probabilities for each word in the message
    for word in words_in_message:
        total_log_prob_spam += spam_log_probs[word]
        total_log_prob_ham += ham_log_probs[word]
    predicted_idx = np.argmax([total_log_prob_spam, total_log_prob_ham])
    return names[predicted_idx]


if __name__ == "__main__":
    ### Read arguments
    if len(sys.argv) != 4:
        print(USAGE % sys.argv[0])
    testing_folder = sys.argv[1]
    (spam_folder, ham_folder) = sys.argv[2:4]

    ### Learn the distributions
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
    (log_probabilities_by_category, log_priors_by_category) = learn_distributions(
        file_lists
    )

    # Here, columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    performance_measures = np.zeros([2, 2])

    ### Classify and measure performance
    for filename in util.get_files_in_folder(testing_folder):
        ## Classify
        label = classify_message(
            filename,
            log_probabilities_by_category,
            log_priors_by_category,
            ["spam", "ham"],
        )
        ## Measure performance
        # Use the filename to determine the true label
        base = os.path.basename(filename)
        true_index = "ham" in base
        guessed_index = label == "ham"
        performance_measures[int(true_index), int(guessed_index)] += 1

        # Uncomment this line to see which files your classifier
        # gets right/wrong:
        # print("%s : %s" %(label, filename))

    template = "You correctly classified %d out of %d spam messages, and %d out of %d ham messages."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0], totals[0], correct[1], totals[1]))
