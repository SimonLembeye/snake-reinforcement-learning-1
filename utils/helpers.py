import json

def export_q_table(q_dict, file_name):
    q_dict_json = {}
    for state in q_dict.keys():
        q_dict_json[str(state)] = {}
        for action in q_dict[state].keys():
            q_dict_json[str(state)][str(action)] = q_dict[state][action]
    with open(file_name + ".json","w") as fp:
        json.dump(q_dict_json,fp, skipkeys=True)
    print("[!] Dumped Q-table to JSON file")
