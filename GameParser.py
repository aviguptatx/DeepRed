import json
import random

def populate_inputs(file_name):
    with open(file_name) as f:
        data = json.load(f)

        # Check not custom
        if data["customGameSettings"]["enabled"]:
            return None, None
        # Check if the game is 7 players
        if len(data["players"]) != 7:
            return None, None
        # Check not rebalanced 7p
        if data["gameSetting"]["rebalance7p"]:
            return None, None

        # Length 7 = (1-7 roles)
        roles = []
        # Length 7 = (1-7 roles)
        my_role = []
        # Lenth 228 = 19 (1-7 - pres, 8-14 chanc, 15 pres claim, 16 chanc claim, 17 policy not enacted, 18 blue, 19 red) * 12 (# possible govs, including VZ)
        gov_data = []
        # Length 15 (1-7 - pres seat number) (8-14 - chancellor seat number) (15 - result)
        investigation_data = []
        # Length 14 (1-7 - pres seat number) (1-7 - chancellor seat number)
        special_election_data = []
        # Length 14 (1-7 - pres seat number) (1-7 - shot seat number)
        bullet_data_1 = []
        # Length 14 (1-7 - pres seat number) (1-7 - shot seat number)
        bullet_data_2 = []
        # Was hitler elected
        # hitler_elected = false

        # Roles
        for seat in range(0, 7):
            roles.append(1 if (data["players"][seat]["role"] == "fascist" or data["players"][seat]["role"] == "hitler") else 0)

        # Pick one of the liberal
        randomLib = random.randint(0, 4)
        libCount = 0
        confirmedSeat = 0
        for seat in range(0, 7):
            if data["players"][seat]["role"] == "liberal":
                if libCount == randomLib:
                    confirmedSeat = seat
                libCount = libCount + 1

        # Append the the my_role array to one-hot encode which seat the AI is playing
        for seat in range(0, 7):
            my_role.append(seat == confirmedSeat)
                
    
        # For each government
        for gov in range(0, len(data["logs"])):
            # If the government was played
            if len(data["logs"][gov]) >= 8:
                # President seat number
                for pres in range(0, 7):
                    gov_data.append(1 if data["logs"][gov]["presidentId"] == pres else 0)
                    
                # Chancellor seat number
                for chan in range(0, 7):
                    gov_data.append(1 if data["logs"][gov]["chancellorId"] == chan else 0)

                # President number of reds claimed
                if "presidentClaim" in data["logs"][gov]: 
                    gov_data.append(data["logs"][gov]["presidentClaim"]["reds"])
                elif "chancellorClaim" in data["logs"][gov]:
                    gov_data.append(data["logs"][gov]["chancellorClaim"]["reds"] + 1)
                elif "enactedPolicy" in data["logs"][gov]:
                    gov_data.append(3 if data["logs"][gov]["enactedPolicy"] == "fascist" else 2)
                else:
                    gov_data.append(3)

                # Chancellor number of reds claimed
                if "chancellorClaim" in data["logs"][gov]:
                    gov_data.append(data["logs"][gov]["chancellorClaim"]["reds"])
                elif "enactedPolicy" in data["logs"][gov]:
                    gov_data.append(2 if data["logs"][gov]["enactedPolicy"] == "fascist" else 1)
                else:
                    gov_data.append(2)
                
                # No policy enacted?
                gov_data.append(not "enactedPolicy" in data["logs"][gov])

                # Red enacted?, blue enacted?
                if "enactedPolicy" in data["logs"][gov]:
                    gov_data.append(0 if data["logs"][gov]["enactedPolicy"] == "fascist" else 1)
                    gov_data.append(1 if data["logs"][gov]["enactedPolicy"] == "fascist" else 0)
                else:
                    gov_data.append(0);
                    gov_data.append(0);
                
                # If investigation
                if("investigationId" in data["logs"][gov]):
                    for pres in range(0, 7):
                        investigation_data.append(1 if data["logs"][gov]["presidentId"] == pres else 0)
                    for chan in range(0, 7):
                        investigation_data.append(1 if data["logs"][gov]["investigationId"] == chan else 0)
                    if not "investigationClaim" in data["logs"][gov]:
                        return None, None
                        investigation_data.append(1 if data["logs"][gov]["investigationClaim"] == "fascist" else 0)

                # If Special Election
                if("specialElection" in data["logs"][gov]):
                    for pres in range(0, 7):
                        special_election_data.append(1 if data["logs"][gov]["presidentId"] == pres else 0)
                    for chan in range(0, 7):
                        special_election_data.append(1 if data["logs"][gov]["specialElection"] == chan else 0)
                # If Bullet
                if("execution" in data["logs"][gov]):
                    # If first bullet
                    if (len(bullet_data_1) < 1):
                        for pres in range(0, 7):
                            bullet_data_1.append(1 if data["logs"][gov]["presidentId"] == pres else 0)
                        for shot in range(0, 7):
                            bullet_data_1.append(1 if data["logs"][gov]["execution"] == shot else 0)
                    # If second bullet
                    else:
                        for pres in range(0, 7):
                            bullet_data_2.append(1 if data["logs"][gov]["presidentId"] == pres else 0)
                        for shot in range(0, 7):
                            bullet_data_2.append(1 if data["logs"][gov]["execution"] == shot else 0)

        for i in range(len(gov_data), 228):
            gov_data.append(0)
        for i in range(len(investigation_data), 15):
            investigation_data.append(0)
        for i in range(len(special_election_data), 14):
            special_election_data.append(0)
        for i in range(len(bullet_data_1), 14):
            bullet_data_1.append(0)
        for i in range(len(bullet_data_2), 14):
            bullet_data_2.append(0)

        game_data = gov_data + investigation_data + special_election_data + bullet_data_1 + bullet_data_2 + my_role
        return game_data, roles
