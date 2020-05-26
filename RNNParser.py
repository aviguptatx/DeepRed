import json
import random

def populate_inputs(game_number_start, game_number_end, lib_inc):
    games = []
    results = []
    test_game_numbers = []

    # games[X][Y][Z] = [Game #][Gov #][Node in gov]
    # results[X][Y] = results[Game #][Seat #]

    for game_number in range(game_number_start, game_number_end):
        game_valid = True
        game_data = []

        # Length 7 = (1-7 roles)
        roles = []
        # Length 7: 1 denotes the lib seat the AI is
        my_role = []

        file_name = "games/" + str(game_number) + ".json"

        with open(file_name) as f:
            data = json.load(f)

            # Check not custom
            if data["customGameSettings"]["enabled"]:
                continue
            # Check if the game is 7 players
            if len(data["players"]) != 7:
                continue
            # Check not rebalanced 7p
            if data["gameSetting"]["rebalance7p"]:
                continue

            # Roles
            for seat in range(0, 7):
                roles.append(1 if (data["players"][seat]["role"] == "fascist" or data["players"][seat]["role"] == "hitler") else 0)

            # Pick one of the liberals psuedorandomly (lib_inc is parameterized)
            random_lib = (game_number + lib_inc) % 4
            lib_count = 0
            confirmed_seat = 0
            for seat in range(0, 7):
                if data["players"][seat]["role"] == "liberal":
                    if lib_count == random_lib:
                        confirmed_seat = seat
                    lib_count = lib_count + 1

            # Append the the my_role array to one-hot encode which seat the AI is playing
            for seat in range(0, 7):
                my_role.append(1 if seat == confirmed_seat else 0)

            if len(data["logs"]) <= 4:
                game_valid = False

            # For each government
            for gov in range(0, len(data["logs"])):

                appending = False
                # Lenth 24 (1-7 - pres, 8-14 chanc, 15-18 pres claim, 19-21 chanc claim, 22 policy not enacted, 23 blue, 24 red)
                gov_data = []
                # Length 15 (1-7 - pres seat number) (8-14 - chancellor seat number) (15 - result)
                investigation_data = []
                # Length 14 (1-7 - pres seat number) (1-7 - chancellor seat number)
                special_election_data = []
                # Length 14 (1-7 - pres seat number) (1-7 - shot seat number)
                bullet_data_1 = []
                # Length 14 (1-7 - pres seat number) (1-7 - shot seat number)
                bullet_data_2 = []
                # Length 3 - (1 - No topdeck, 2 - Topdeck blue, 3 - Topdeck red)
                topdecks = []

                # If the government was played
                if len(data["logs"][gov]) >= 7:
                    appending = True

                    # President seat number
                    for pres in range(0, 7):
                        gov_data.append(1 if data["logs"][gov]["presidentId"] == pres else 0)

                    # Chancellor seat number
                    for chan in range(0, 7):
                        gov_data.append(1 if data["logs"][gov]["chancellorId"] == chan else 0)

                    pres_claim = data["logs"][gov]["presidentClaim"]["reds"] if "presidentClaim" in data["logs"][gov] else 0
                    chanc_claim = data["logs"][gov]["chancellorClaim"]["reds"] if "chancellorClaim" in data["logs"][gov] else 0

                    # President number of reds claimed
                    if "presidentClaim" in data["logs"][gov]:
                        gov_data.append(1 if pres_claim == 0 else 0)
                        gov_data.append(1 if pres_claim == 1 else 0)
                        gov_data.append(1 if pres_claim == 2 else 0)
                        gov_data.append(1 if pres_claim == 3 else 0)
                    elif "chancellorClaim" in data["logs"][gov]:
                        gov_data.append(0)
                        gov_data.append(1 if chanc_claim == 0 else 0)
                        gov_data.append(1 if chanc_claim == 1 else 0)
                        gov_data.append(1 if chanc_claim == 2 else 0)
                    elif "enactedPolicy" in data["logs"][gov]:
                        game_valid = False
                    else:
                        game_valid = False

                    # Chancellor number of reds claimed
                    if "chancellorClaim" in data["logs"][gov]:
                        gov_data.append(1 if chanc_claim == 0 else 0)
                        gov_data.append(1 if chanc_claim == 1 else 0)
                        gov_data.append(1 if chanc_claim == 2 else 0)
                    elif "enactedPolicy" in data["logs"][gov]:
                        gov_data.append(0)
                        gov_data.append(
                            0 if data["logs"][gov]["enactedPolicy"] == "fascist" else 1)
                        gov_data.append(
                            1 if data["logs"][gov]["enactedPolicy"] == "fascist" else 0)
                    else:
                        game_valid = False

                    # No policy enacted?
                    gov_data.append(0 if "enactedPolicy" in data["logs"][gov] else 1)

                    # Red enacted?, blue enacted?
                    if "enactedPolicy" in data["logs"][gov]:
                        gov_data.append(
                            0 if data["logs"][gov]["enactedPolicy"] == "fascist" else 1)
                        gov_data.append(
                            1 if data["logs"][gov]["enactedPolicy"] == "fascist" else 0)
                    else:
                        gov_data.append(0)
                        gov_data.append(0)

                    # If investigation
                    if "investigationId" in data["logs"][gov]:
                        for pres in range(0, 7):
                            investigation_data.append(
                                1 if data["logs"][gov]["presidentId"] == pres else 0)
                        for chan in range(0, 7):
                            investigation_data.append(
                                1 if data["logs"][gov]["investigationId"] == chan else 0)
                        if not "investigationClaim" in data["logs"][gov]:
                            game_valid = False
                        else:
                            investigation_data.append(
                                1 if data["logs"][gov]["investigationClaim"] == "fascist" else 0)

                    # If Special Election
                    if "specialElection" in data["logs"][gov]:
                        for pres in range(0, 7):
                            special_election_data.append(
                                1 if data["logs"][gov]["presidentId"] == pres else 0)
                        for chan in range(0, 7):
                            special_election_data.append(
                                1 if data["logs"][gov]["specialElection"] == chan else 0)
                    # If Bullet
                    if "execution" in data["logs"][gov]:
                        # If first bullet
                        if (len(bullet_data_1) < 1):
                            for pres in range(0, 7):
                                bullet_data_1.append(
                                    1 if data["logs"][gov]["presidentId"] == pres else 0)
                            for shot in range(0, 7):
                                bullet_data_1.append(
                                    1 if data["logs"][gov]["execution"] == shot else 0)
                        # If second bullet
                        else:
                            for pres in range(0, 7):
                                bullet_data_2.append(
                                    1 if data["logs"][gov]["presidentId"] == pres else 0)
                            for shot in range(0, 7):
                                bullet_data_2.append(
                                    1 if data["logs"][gov]["execution"] == shot else 0)

                # If the government was a topdeck
                if len(data["logs"][gov]) == 4 and ("enactedPolicy" in data["logs"][gov]):
                    appending = True
                    topdecks.append(0)
                    topdecks.append(
                        0 if data["logs"][gov]["enactedPolicy"] == "fascist" else 1)
                    topdecks.append(
                        1 if data["logs"][gov]["enactedPolicy"] == "fascist" else 0)
                else:
                    topdecks.append(1)
                    topdecks.append(0)
                    topdecks.append(0)

                for i in range(len(gov_data), 24):
                    gov_data.append(0)
                for i in range(len(investigation_data), 15):
                    investigation_data.append(0)
                for i in range(len(special_election_data), 14):
                    special_election_data.append(0)
                for i in range(len(bullet_data_1), 14):
                    bullet_data_1.append(0)
                for i in range(len(bullet_data_2), 14):
                    bullet_data_2.append(0)

                if appending:
                    game_data.append(gov_data + investigation_data + special_election_data + bullet_data_1 + bullet_data_2 + topdecks + my_role)

            for i in range(len(game_data), 12):
                game_data.append([0] * 91)

            if (game_valid):
                games.append(game_data)
                results.append(roles)
                test_game_numbers.append(game_number)

    return games, results, test_game_numbers