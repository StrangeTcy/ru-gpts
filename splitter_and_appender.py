from sklearn.model_selection import train_test_split

print ("Reading original datasets...")

with open("/home/enoch/ivolution/programs_training.txt", "r") as f:
	orig_train = f.read()

with open("/home/enoch/ivolution/programs_eval.txt", "r") as f:
	orig_eval = f.read()	

with open("/home/enoch/ivolution/programs_training.json", "r") as f:
	orig_train_json = f.read()

with open("/home/enoch/ivolution/programs_eval.json", "r") as f:
	orig_eval_json = f.read()


print ("Reading additional datasets...")

with open("/home/enoch/ivolution/from_laptop/programs_training.txt", "r") as f:
	more_train = f.read()

with open("/home/enoch/ivolution/from_laptop/programs_training.json", "r") as f:
	more_train_json = f.read()

print ("Splitting...")

append_train_json, append_eval_json, append_train, append_eval = \
		train_test_split(more_train_json, more_train, test_size=0.33, shuffle=False)


print ("Trying to write back...")

with open("/home/enoch/ivolution/programs_training.txt", "a") as f:
	f.write(append_train)

with open("/home/enoch/ivolution/programs_eval.txt", "a") as f:
	f.write(append_eval)	

with open("/home/enoch/ivolution/programs_training.json", "a") as f:
	f.write(append_train_json)

with open("/home/enoch/ivolution/programs_eval.json", "a") as f:
	f.write(append_eval_json)
