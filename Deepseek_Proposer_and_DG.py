from openai import OpenAI
import os
import google.generativeai as genai
import statistics
import math
import scipy.stats as stats
import numpy as np
import csv


def run(id, stakes_value, wealth_value, model_value, temperature_value, framing_value, human_emphasis):
    #Gemini Set-up
    global id_counter
    global proposals
    global responses
    global proposal_error
    global responder_error
    global master_list
    global aggregate_master_list
    global model
    global reasoning_counter
    global wealth
    global stakes

    repetitions = 100
    wealth_int = wealth_value
    wealth = str(wealth_int)
    stakes = str(stakes_value)
    int_stakes = stakes_value
    proposals = []
    responses = []
    proposal_error = 0
    responder_error = 0
    reasoning_counter = 1


    prompt = "Respond to the following prompt as if you are human. Before participating in this experiment you have " + wealth + " Rupees in wealth. Welcome to this study of decision-making. The experiment will take about 15 minutes. The instructions are simple, and if you follow them carefully, you can earn a considerable amount of money. All the money you earn is yours to keep, and will be paid to you, in cash, in private, after the experiment ends. Your confidentiality is assured. In this experiment, you have been assigned the role of 'proposer.' You have been randomly matched with another participant who will be in the role of 'responder.' Your earnings will depend on your decisions, as well as on the decisions of the responder. You will be asked to propose a split of a total of " + stakes + " Rupees between yourself and the responder. That is, you will make an offer to the responder that specifies how much of the " + stakes + " Rupees you will receive and how much of the " + stakes + " Rupees he/she will receive. The amount that your offer specifies for yourself can be anything from 0 to " + stakes + " Rupees. Your earnings in the experiment will depend on whether or not the responder accepts your offer. If he/she accepts your offer, both you and the responder receive the amounts specified in your (accepted) offer. If he/she rejects your offer, both you and the responder will receive zero earnings for this experiment. Notice that if the responder's goal is to earn as much money as possible from the experiment, he/she should accept any offer that gives him/her positive earnings, no matter how low. This is because the alternative is reject, in which he/she will not receive any earnings. If the responder is expected to behave in this way and accept any positive offer, a proposer should offer the minimum possible amount to the responder in order to leave the experiment with as much money as possible. That is, if the responder that you are matched with aims to earn as much money as possible, he/she should accept any offer that is greater than zero. Given this, making the offer that gives the lowest possible earnings to the responder will allow you to leave the experiment with as much money possible. Now, please tell us your proposed split of the " + stakes + " Rupees between yourself and the responder. You must only respond with the numerical value of what you propose to give to the responder and nothing else." #Insert prompt here
    for x in range(repetitions):
        get_ai_response(prompt, model_value, "Data.txt")
        reasoning_counter += 1
    int_average = takeaverage("Data.txt")
    os.remove("Data.txt")
    avg_stake_percentage = int_average/int_stakes * 100
    zero_proposals = 0
    for p in proposals:
        if p == 0:
            zero_proposals += 1

    proposer_std_dev = statistics.stdev(proposals)
    proposer_sample_size = len(proposals)
    proposer_std_err = proposer_std_dev / math.sqrt(proposer_sample_size)
    proposer_t_critical = stats.t.ppf((1 + 0.95) / 2, df=proposer_sample_size - 1)
    proposer_margin_of_error = proposer_t_critical * (proposer_std_err / np.sqrt(proposer_sample_size))
    proposer_ci_lower = int_average - proposer_margin_of_error
    proposer_ci_upper = int_average + proposer_margin_of_error

    for i in range(proposer_sample_size):
        individual_list = []
        individual_list.append(proposals[i])
        individual_list.append(id)
        individual_list.append(stakes_value)
        individual_list.append(wealth_value)
        individual_list.append(model_value)
        proportion_value = proposals[i] / stakes_value
        individual_list.append(proportion_value)
        individual_list.append(temperature_value)
        individual_list.append("March")
        individual_list.append(framing_value)
        individual_list.append(human_emphasis)
        
        master_list.append(individual_list)
    
    aggregate_individual_list = []
    aggregate_individual_list.append(id)
    aggregate_individual_list.append(model_value)
    aggregate_individual_list.append(stakes_value)
    aggregate_individual_list.append(wealth_value)
    aggregate_individual_list.append(temperature_value)
    aggregate_individual_list.append("March")
    aggregate_individual_list.append(int_average)
    aggregate_individual_list.append(avg_stake_percentage)
    aggregate_individual_list.append(zero_proposals)
    aggregate_individual_list.append(proposer_std_err)
    aggregate_individual_list.append(proposer_ci_lower)
    aggregate_individual_list.append(proposer_ci_upper)
    aggregate_individual_list.append(proposal_error)
    aggregate_individual_list.append(framing_value)
    aggregate_individual_list.append(human_emphasis)
    aggregate_master_list.append(aggregate_individual_list)

    id_counter += 1

def chat_with_gpt(prompt):
    global client
    global reasoning_counter
    try:
        response = client.chat.completions.create(
            model= "deepseek-reasoner",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=150
        )
        reasoning_content = response.choices[0].message.reasoning_content
        str_reasoning_counter = str(reasoning_counter)
        reasoning_return = open("", 'a') #file to save reasoning content
        reasoning_return.write(str_reasoning_counter + '. ' + "Stakes: " + stakes + " Wealth: " + wealth + '\n' + reasoning_content + '\n \n \n')
        reasoning_return.close()
        return response.choices[0].message.content
    except Exception as e:
        print("Error:", e)
        return None    

def get_ai_response(prompt, model_value, text_file):
    while True:
        response = chat_with_gpt(prompt)
        if response:
            data = open(text_file, 'a')
            data.write(response + '\n')
            data.close()
            return response
        else:
            print("Failed to get response from the AI 1")

def isnumber(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def takeaverage(file):
    global proposal_error
    global proposals
    total = 0
    count = 0
    with open(file, 'r') as file:
        for l in file:
            l = l.strip()
            if isnumber(l):
                total += float(l)
                count += 1
                int_l = int(l)
                proposals.append(int_l)
            else:
                print("Rejected non-numeric value")
                proposal_error += 1
    return total/count

def average_responses(file):
    global responder_error
    global responses
    total_zeroes = 0
    total_ones = 0
    with open(file, 'r') as file:
        for line in file:
            one_count = 0
            zero_count = 0
            line = line.strip()
            one_count += line.count('yes')
            one_count += line.count('Yes')
            zero_count += line.count('no')
            zero_count += line.count('No')
            if zero_count >= 1:
                total_zeroes += 1
                responses.append(0)
            elif zero_count == 0:
                total_ones += 1
                responses.append(1)
            else:
                responses.append(' ')
                responder_error += 1
                print("ERROR")
    ratio = total_ones / (total_zeroes + total_ones)
    return ratio

#OpenAI setup
client = OpenAI(api_key='', base_url="https://api.deepseek.com") #Insert API Key Here

id_counter = 1
master_list = []
column_list = ["Proposal", "ID", "Stakes", "Wealth", "Model", "Proportion", "Temperature", "Month", "Framing", "Human_Emphasis"]
master_list.append(column_list)
aggregate_master_list = [] 
aggregate_column_list = ["ID", "Model", "Stakes (Rupees)", "Wealth (Rupees)", "Temperature", "Month", "Prop_Avg (Rupees)", "Avg_Per_of_Stake", "Num_of_Zero_Offers", "Prop_SE", "Prop_CI_95_L", "Prop_CI_95_H", "Prop_Errors", "Framing", "Human_Emphasis"]
aggregate_master_list.append(aggregate_column_list)
proposals = []
responses = []
proposal_error = 0
responder_error = 0

def main():
    global master_list
    global aggregate_master_list

    #run(id_counter, 20, 493, "Deepseek-Reasoner", 1, 3, 0)

    file_path_one = 'Deepseek_UG_Proposer.csv'
    file_path_two = 'Deepseek_UG_Proposer_Aggregate.csv'

    file_exists_one = os.path.isfile(file_path_one)
    file_exists_two = os.path.isfile(file_path_two)

    with open(file_path_one, mode='a', newline='') as file:
        writer = csv.writer(file)
    
        if not file_exists_one:
            writer.writerow(master_list[0])
    
        writer.writerows(master_list[1:])
    
    with open(file_path_two, mode='a', newline='') as file_two:
        writer_2 = csv.writer(file_two)

        if not file_exists_two:
            writer_2.writerow(aggregate_master_list[0])
        
        writer_2.writerows(aggregate_master_list[1:])
        

        



    


main()