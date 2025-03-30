from openai import OpenAI
import os
import google.generativeai as genai
import statistics
import math
import scipy.stats as stats
import numpy as np
import csv


def run(id, stakes_value, wealth_value, model_value, temperature_value, proportion_value, framing, human_emphasis):
    #Gemini Set-up
    global id_counter
    global proposals
    global responses
    global proposal_error
    global responder_error
    global master_list
    global aggregate_master_list
    global model
    genai.configure(api_key='') #Insert API Key Here
    repetitions = 150
    wealth_int = wealth_value

    wealth = str(wealth_int)
    stakes = str(stakes_value)
    int_stakes = stakes_value
    proposals = []
    responses = []
    proposal_error = 0
    responder_error = 0

    proportion = proportion_value * int_stakes
    for x in range(repetitions):
        proposals.append(proportion)
    int_average = proportion
    for p in proposals:
        int_proposal = p
        proposal = str(int_proposal)
        int_remainder = int_stakes - int_proposal
        remainder = str(int_remainder)
        responder_prompt = "Respond to the following prompt as if you are human. Before participating you have " + wealth + " Rupees in wealth. Welcome to this study of decision-making. The experiment will take about 15 minutes. The instructions are simple, and if you follow them carefully, you can earn a considerable amount of money. All the money you earn is yours to keep, and will be paid to you, in cash, in private, after the experiment ends. Your confidentiality is assured. In this experiment, you have been assigned the role of 'responder.' You have been randomly matched with another participant who is in the role of 'proposer.' Your earnings will depend on your decisions, as well as on the decision of the proposer. The proposer has been asked to propose a split of " + stakes + " Rupees between him/her and you. That is, the proposer has made an offer that specifies how much of the total " + stakes + " Rupees you will recieve and how much of the " + stakes + " Rupees he/she will recieve. You can choose either to accept or reject this offer. If you accept the offer, both you and the proposer recieve the amounts specified in the offer. If you reject the offer, both you and the proposer will recieve zero earnings from this experiment. The proposer has offered that out of the total amount of " + stakes + " Rupees, you recieve " +  proposal + " Rupees and he/she recieves " + remainder + " Rupees. Now, please tell us if you accept or decline this offer by the proposer. You must only respond with 'yes' if you accept the offer and only respond with 'no' if you reject the offer." #Insert Prompt Here
        get_ai_response(responder_prompt, "Responses.txt", model_value, temperature_value)
    int_responder_average = average_responses("Responses.txt") 
    os.remove("Responses.txt")
    avg_stake_percentage = int_average/int_stakes * 100
    zero_proposals = 0
    for p in proposals:
        if p == 0:
            zero_proposals += 1
    percent_responder_average = int_responder_average * 100

    proposer_std_dev = statistics.stdev(proposals)
    proposer_sample_size = len(proposals)
    proposer_std_err = proposer_std_dev / math.sqrt(proposer_sample_size)
    proposer_t_critical = stats.t.ppf((1 + 0.95) / 2, df=proposer_sample_size - 1)
    proposer_margin_of_error = proposer_t_critical * (proposer_std_err / np.sqrt(proposer_sample_size))
    proposer_ci_lower = int_average - proposer_margin_of_error
    proposer_ci_upper = int_average + proposer_margin_of_error

    responder_std_dev = statistics.stdev(responses)
    responder_sample_size = len(responses)
    responder_std_err = responder_std_dev / math.sqrt(responder_sample_size)
    responder_t_critical = stats.t.ppf((1 + 0.95) / 2, df=responder_sample_size - 1)
    responder_margin_of_error = responder_t_critical * (responder_std_err / np.sqrt(responder_sample_size))
    responder_ci_lower = int_responder_average - responder_margin_of_error
    responder_ci_upper = int_responder_average + responder_margin_of_error

    for i in range(proposer_sample_size):
        individual_list = []
        individual_list.append(proposals[i])
        individual_list.append(responses[i])
        individual_list.append(id)
        individual_list.append(stakes_value)
        individual_list.append(wealth_value)
        individual_list.append(model_value)
        proportion_value = proposals[i] / stakes_value
        individual_list.append(proportion_value)
        individual_list.append(temperature_value)
        individual_list.append("March")
        individual_list.append(framing)
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
    aggregate_individual_list.append(percent_responder_average)
    aggregate_individual_list.append(responder_std_err)
    aggregate_individual_list.append(responder_ci_lower)
    aggregate_individual_list.append(responder_ci_upper)
    aggregate_individual_list.append(responder_error)
    aggregate_individual_list.append(framing)
    aggregate_individual_list.append(human_emphasis)
    aggregate_master_list.append(aggregate_individual_list)

    id_counter += 1

def chat_with_gpt(prompt, model_number, temperature_cond):
    global client
    if model_number == "GPT-3.5":
        model_type = "gpt-3.5-turbo"
    elif model_number == "GPT-4":
        model_type = "gpt-4"
    else:
        model_type = "gpt-4o"
    if temperature_cond == 0:
        temperature_setting = 0
    else:
        temperature_setting = 1
    try:
        response = client.chat.completions.create(
            model=model_type,
            temperature=temperature_setting,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Error:", e)
        return None

def chat_with_gemini(prompt):
    global model
    try:
        response = model.generate_content(prompt)
        text_response = response.text.strip()
        return text_response
    except Exception as e:
        print("Error:", e)
        return None

def get_ai_response(prompt, text_file, model, temperature):
    while True:
            response = chat_with_gpt(prompt, model, temperature)
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
client = OpenAI(api_key='') #Insert API Key Here

model = genai.GenerativeModel('gemini-1.5-pro')

id_counter = 1
master_list = []
column_list = ["Proposal", "Response", "ID", "Stakes", "Wealth", "Model", "Proportion", "Temperature", "Month", "Framing", "Human_Emphasis"]
master_list.append(column_list)
aggregate_master_list = [] 
aggregate_column_list = ["ID", "Model", "Stakes (Rupees)", "Wealth (Rupees)", "Temperature", "Month", "Prop_Avg (Rupees)", "Avg_Per_of_Stake", "Num_of_Zero_Offers", "Prop_SE", "Prop_CI_95_L", "Prop_CI_95_H", "Prop_Errors", "Resp_Accep_Rate (%)", "Resp_SE", "Resp_CI_95_L", "Resp_CI_95_H", "Resp_Errors", "Framing", "Human_Emphasis"]
aggregate_master_list.append(aggregate_column_list)
proposals = []
responses = []
proposal_error = 0
responder_error = 0

def main():
    global master_list
    global aggregate_master_list

    #run(id_counter, 20, 493, "GPT-4o", 1, 0.1, 1, 0)



    file_path_one = ''
    file_path_two = ''

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