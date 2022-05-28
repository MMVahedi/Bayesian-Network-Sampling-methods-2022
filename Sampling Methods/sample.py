# Name: Mohammad Mahdi Vahedi
# Student number: 99109314

import json
import os

import numpy as np
from matplotlib import pyplot as plt


def increase_binary(number):
    counter = 0
    while True:
        if number[counter] == 0:
            number[counter] = 1
            return number
        else:
            number[counter] = 0
            counter += 1


def update_cpt(node, value, cpt):
    new_cpt = []
    for row in cpt:
        if row[node] == value:
            new_cpt.append(row)
    return new_cpt


def find_distribution(query_variables, cpts, n, evidence_nodes):
    order = [i for i in range(n)]
    for query_var in query_variables:
        order.remove(query_var)
    for evidence in evidence_nodes:
        order.remove(evidence)
    for node in order:
        cpts, joint_cpts = find_cpts_with_node(node, cpts)
        joint_cpt = find_joint(joint_cpts)
        eliminated_cpt = eliminate(node, joint_cpt)
        cpts.append(eliminated_cpt)
    return cpts


def eliminate(node, joint_cpt):
    dict = {}
    new_cpt = []
    for row in joint_cpt:
        token = ""
        for key in row.keys():
            if key != node and key != 'P':
                if row[key]:
                    token = token + '1'
                else:
                    token = token + '0'
        if row[node]:
            token = '1' + token
        else:
            token = '0' + token
        row["token"] = token
        dict[token] = row
    for row in joint_cpt:
        token = row["token"]
        if dict[token] != 'N':
            sum = 0
            if token[0] == '0':
                sum = dict['1' + token[1:]]['P']
                dict['1' + token[1:]] = 'N'
            else:
                sum = dict['0' + token[1:]]['P']
                dict['0' + token[1:]] = 'N'
            row['P'] = row['P'] + sum
            del row[node]
            del row["token"]
            new_cpt.append(row.copy())
    return new_cpt


def find_cpts_with_node(node, cpts):
    new_cpts = []
    joint_cpts = []
    for cpt in cpts:
        if node in cpt[0].keys():
            joint_cpts.append(cpt)
        else:
            new_cpts.append(cpt)
    return new_cpts, joint_cpts


def find_joint(cpts):
    if len(cpts) == 1:
        return cpts[0]
    for i in range(1, len(cpts)):
        cpts[0] = multipy_table(cpts[0], cpts[i])
    return cpts[0]


def multipy_table(first, second):
    new_cpt = []
    first_keys = first[0].keys()
    second_keys = second[0].keys()
    join_keys = []
    nonjoint_keys = []
    for key in second_keys:
        if key in first_keys and key != 'P':
            join_keys.append(key)
        else:
            nonjoint_keys.append(key)
    for second_row in second:
        for first_row in first:
            flag = True
            for key in join_keys:
                if first_row[key] != second_row[key]:
                    flag = False
            if flag:
                new_row = first_row.copy()
                for key in nonjoint_keys:
                    if key != 'P':
                        new_row[key] = second_row[key]
                    else:
                        new_row['P'] = first_row['P'] * second_row['P']
                new_cpt.append(new_row)
    return new_cpt


def read_bayes_net(input_string):
    input_string.reverse()
    n = int(input_string.pop())
    bayes_net = [[[], []] for _ in range(n)]
    cpts = [[]] * n
    for j in range(n):
        i = int(input_string.pop())
        bayes_net[i][0] = list(map(lambda x: int(x), input_string.pop().split()))
        for parent in bayes_net[i][0]:
            bayes_net[parent][1].append(i)
        probability = list(map(float, input_string.pop().split()))
        parents = bayes_net[i][0].copy()
        parents.reverse()
        cpt = []
        number = [0] * (len(parents) + 2)
        counter = 0
        while True:
            row = {}
            for k in range(len(parents)):
                if number[k] == 0:
                    row[parents[k]] = True
                else:
                    row[parents[k]] = False
            row[i] = number[len(parents)] == 0
            number = increase_binary(number)
            cpt.append(row)
            if counter < len(probability):
                row['P'] = probability[counter]
            else:
                row['P'] = 1 - probability[counter - len(probability)]
            counter += 1
            if number[len(parents) + 1] == 1:
                break
        cpts[i] = cpt
    return n, bayes_net, cpts


def read_evidence(dict, node_numbers):
    evidence = {}
    for key in dict.keys():
        node = node_numbers[key]
        evidence[node] = dict[key] == 1
    return evidence


def read_query(dict, node_numbers):
    query = {}
    for key in dict.keys():
        node = node_numbers[key]
        query[node] = dict[key] == 1
    return query


def update_net(evidence, cpts, bayes_net):
    for evi in evidence.keys():
        node = evi
        val = evidence[evi]
        for child in bayes_net[node][1]:
            cpts[child] = update_cpt(node, val, cpts[child])
        cpts[node] = update_cpt(node, val, cpts[node])


def find_real_probability(query, evidence, cpts, n):
    cpts = find_distribution(query.keys(), cpts, n, evidence.keys())
    final_joint = find_joint(cpts)
    probability = 0
    sum = 0
    for row in final_joint:
        sum += row['P']
        flag = True
        for key in query.keys():
            if row[key] != query[key]:
                flag = False
        if flag:
            probability = row['P']
    return probability / sum


def change_net_string(net_string, nodes_numbers):
    net_string.reverse()
    new = [net_string.pop()]
    while len(net_string) > 0:
        line = net_string.pop()
        node_number = nodes_numbers[line[0]]
        new.append(str(node_number))
        line = net_string.pop().split()
        if len(line[0]) == 1:
            if ord(line[0]) < 65:
                new.append("")
                new.append(line[0])
            else:
                lst = line
                parents = ""
                for parent in lst:
                    parents += str(nodes_numbers[parent]) + " "
                new.append(parents)
                probability = []
                flag = True
                for i in range(2 ** len(lst)):
                    p = net_string.pop().split()
                    if p[0] == '1' and i == 0:
                        flag = False
                    probability.append(p[-1])
                if flag:
                    probability.reverse()
                new.append(" ".join(probability))
        else:
            new.append("")
            new.append(line[0])
    return new


def child_topological(visited, bayes_net, i, stack):
    visited[i] = True
    for child in bayes_net[i][1]:
        if not visited[child]:
            child_topological(visited, bayes_net, child, stack)
    stack.append(i)


def find_topological_sort(bayes_net, n):
    visited = [False] * n
    stack = []
    for i in range(n):
        if not visited[i]:
            child_topological(visited, bayes_net, i, stack)
    stack.reverse()
    return stack


def make_copy_of_cpts(cpts):
    mycopy = []
    for cpt in cpts:
        cpt_copy = []
        for row in cpt:
            cpt_copy.append(row.copy())
        mycopy.append(cpt_copy)
    return mycopy


def prior_sample(order, cpts, bayes_net, n):
    sample = {}
    for i in range(n):
        node = order[i]
        acceptable_nodes = find_acceptable_nodes(cpts[node], bayes_net[node][0], sample, node)
        uniform_X = np.random.random()
        if uniform_X <= acceptable_nodes[0]['P']:
            sample[node] = True
        else:
            sample[node] = False
    return sample


def find_acceptable_nodes(cpt, parents, sample, node):
    acceptable = [0, 0]
    for row in cpt:
        flag = True
        for parent in parents:
            if sample[parent] != row[parent]:
                flag = False
        if flag:
            if row[node]:
                acceptable[0] = row
            else:
                acceptable[1] = row
    return acceptable


def prior_sampling(query, evidence, cpts, bayes_net, order, n):
    number_of_samples = 1000
    samples = []
    for i in range(number_of_samples):
        sample = prior_sample(order, cpts, bayes_net, n)
        flag = True
        for key in evidence.keys():
            if sample[key] != evidence[key]:
                flag = False
        if flag:
            samples.append(sample)
    number_of_accepted_samples = 0
    for sample in samples:
        flag = True
        for key in query.keys():
            if query[key] != sample[key]:
                flag = False
        if flag:
            number_of_accepted_samples += 1
    if len(samples) == 0:
        return 1
    else:
        return number_of_accepted_samples / len(samples)


def rejection_sampling(query, evidence, cpts, bayes_net, order, n):
    number_of_samples = 1000
    number_of_accepted_samples = 0
    counter = 0
    limit = 1000
    samples = []
    while counter <= limit and number_of_accepted_samples != number_of_samples:
        rejected, sample = rejection_sample(order, cpts, bayes_net, n, evidence)
        if not rejected:
            samples.append(sample)
            number_of_accepted_samples += 1
        counter += 1
    number_of_samples_equal_with_query = 0
    for sample in samples:
        flag = True
        for key in query.keys():
            if query[key] != sample[key]:
                flag = False
        if flag:
            number_of_samples_equal_with_query += 1
    if number_of_accepted_samples == 0:
        return 1
    else:
        return number_of_samples_equal_with_query / number_of_accepted_samples


def rejection_sample(order, cpts, bayes_net, n, evidence):
    sample = {}
    for i in range(n):
        node = order[i]
        acceptable_nodes = find_acceptable_nodes(cpts[node], bayes_net[node][0], sample, node)
        uniform_X = np.random.random()
        if uniform_X <= acceptable_nodes[0]['P']:
            sample[node] = True
        else:
            sample[node] = False
        if node in evidence.keys():
            if sample[node] != evidence[node]:
                return True, sample
    return False, sample


def likelihood_sampling(query, evidence, cpts, bayes_net, order, n):
    number_of_samples = 1000
    samples = []
    weights = []
    for i in range(number_of_samples):
        w, sample = likelihood_sample(evidence, cpts, bayes_net, order, n)
        samples.append(sample)
        weights.append(w)

    acceptable_weights = 0
    all_weights = 0
    for i in range(len(samples)):
        sample = samples[i]
        w = weights[i]
        all_weights += w
        flag = True
        for q in query.keys():
            if sample[q] != query[q]:
                flag = False
        if flag:
            acceptable_weights += w
    return acceptable_weights / all_weights


def likelihood_sample(evidence, cpts, bayes_net, order, n):
    sample = {}
    w = 1
    for i in range(n):
        node = order[i]
        acceptable_nodes = find_acceptable_nodes(cpts[node], bayes_net[node][0], sample, node)
        if node in evidence.keys():
            sample[node] = evidence[node]
            if evidence[node]:
                w *= acceptable_nodes[0]['P']
            else:
                w *= acceptable_nodes[1]['P']
        else:
            uniform_X = np.random.random()
            if uniform_X <= acceptable_nodes[0]['P']:
                sample[node] = True
            else:
                sample[node] = False
    return w, sample


def find_probability_for_gibbs_sampling(cpts, sample, node, n, bayes_net):
    query = {node: True}
    evidence = sample.copy()
    del evidence[node]
    my_copy = make_copy_of_cpts(cpts)
    update_net(evidence, my_copy, bayes_net)
    return find_real_probability(query, evidence, my_copy, n)


def gibbs_sampling(query, evidence, cpts, bayes_net, order, n):
    x, starting_sample = likelihood_sample(evidence, cpts, bayes_net, order, n)
    number_of_samples = 1000
    samples = []
    for i in range(number_of_samples):
        gibbs_sample(evidence, cpts, bayes_net, order, n, starting_sample)
        samples.append(starting_sample.copy())
    number_of_accepted_samples = 0
    for sample in samples:
        flag = True
        for key in query.keys():
            if query[key] != sample[key]:
                flag = False
        if flag:
            number_of_accepted_samples += 1
    return number_of_accepted_samples / len(samples)


def gibbs_sample(evidence, cpts, bayes_net, order, n, sample):
    for i in range(n):
        node = order[i]
        if not (node in evidence.keys()):
            probability = find_probability_for_gibbs_sampling(cpts, sample, node, n, bayes_net)
            uniform_X = np.random.random()
            if uniform_X <= probability:
                sample[node] = True
            else:
                sample[node] = False


def find_node_numbers(input_lines):
    n = int(input_lines[0])
    dict = {}
    index = 1
    for i in range(n):
        node = input_lines[index]
        dict[node[0]] = i
        line = input_lines[index + 1].split()
        if len(line[0]) == 1:
            if ord(line[0]) < 65:
                index += 2
            else:
                index += 2 + 2 ** len(line)
        else:
            index += 2
    return dict


def make_plot(prior, rejection, likelihood, gibbs, path, index):
    X = [i + 1 for i in range(len(prior))]
    plt.plot(X, prior, color='r', label='Prior')
    plt.plot(X, rejection, color='g', label='Rejection')
    plt.plot(X, likelihood, color='y', label='Likelihood')
    plt.plot(X, gibbs, color='b', label='Gibbs')
    plt.xlabel("Query number")
    plt.ylabel("MAE")
    plt.legend()
    path = os.path.join(path, str(index) + ".png")
    plt.savefig(path)
    plt.show()


def run_program_for_simple_input(input_path, index, output_path):
    f = open(os.path.join(input_path, "input.txt"), "r")
    input_lines = f.readlines()
    f.close()
    for i in range(len(input_lines)):
        if input_lines[i] == "\n":
            input_lines = input_lines[:i]
            break
        else:
            input_lines[i] = input_lines[i][:-1]
    nodes_numbers = find_node_numbers(input_lines)
    new_input = change_net_string(input_lines, nodes_numbers)
    n, bayes_net, cpts = read_bayes_net(new_input)

    f = open(os.path.join(input_path, "q_input.txt"), "r")
    input_query = json.load(f)
    f.close()

    text = ""
    prior_difference = []
    rejection_difference = []
    likelihood_difference = []
    gibbs_difference = []
    for q in input_query:
        query = read_query(q[0], nodes_numbers)
        evidence = read_evidence(q[1], nodes_numbers)
        my_copy = make_copy_of_cpts(cpts)
        update_net(evidence, my_copy, bayes_net)
        real_probability = find_real_probability(query, evidence, my_copy, n)
        text += str(round(real_probability, 5)) + " "
        order = find_topological_sort(bayes_net, n)
        my_copy = make_copy_of_cpts(cpts)
        prior_probability = prior_sampling(query, evidence, my_copy, bayes_net, order, n)
        text += str(round(abs(prior_probability - real_probability), 5)) + " "
        prior_difference.append(abs(prior_probability - real_probability))
        my_copy = make_copy_of_cpts(cpts)
        rejection_probability = rejection_sampling(query, evidence, my_copy, bayes_net, order, n)
        text += str(round(abs(rejection_probability - real_probability), 5)) + " "
        rejection_difference.append(abs(rejection_probability - real_probability))
        my_copy = make_copy_of_cpts(cpts)
        likelihood_probability = likelihood_sampling(query, evidence, my_copy, bayes_net, order, n)
        text += str(round(abs(likelihood_probability - real_probability), 5)) + " "
        likelihood_difference.append(abs(likelihood_probability - real_probability))
        my_copy = make_copy_of_cpts(cpts)
        gibbs_probability = gibbs_sampling(query, evidence, my_copy, bayes_net, order, n)
        text += str(round(abs(gibbs_probability - real_probability), 5))
        gibbs_difference.append(abs(gibbs_probability - real_probability))
        text += "\n"
    make_output_file(text, output_path, index)
    make_plot(prior_difference, rejection_difference, likelihood_difference, gibbs_difference, output_path, index)


def make_output_file(text, path, index):
    path = os.path.join(path, str(index) + ".txt")
    f = open(path, "w")
    f.write(text)
    f.close()


# Make Directory
path = os.path.join(os.getcwd(), "output")
output_path = path
try:
    os.mkdir(path)
except OSError as error:
    print("File Already Exist!")

path = os.path.join(os.getcwd(), "inputs")
dirs = os.listdir(path)
for i in range(len(dirs)):
    dir = dirs[i]
    run_program_for_simple_input(os.path.join(path, dir), i + 1, output_path)
