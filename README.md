## DEEP LEARNING PROJECT 
### DTU 02456 DEEP LEARNING 
#### Sanem Leblebici - s222448
#### Michal Lehwark - s222999
#### Ari Menachem - s163956
#### Elli Georgiou - s223408


### INITIAL SETUP NOTES

First run the following command on your command line using your student id\
`ssh <student_id>@login.hpc.dtu.dk`

Then load modules by running the following commands\
`module load python3/3.10.12`\
`module load cuda/12.1`

Then check if the modules loaded correctly by running the following command (you should see python and cuda listed)\
`module list`

Then create a directory in your environment \
run `ls` to see available directories and then run `cd Desktop` to go into desktop\
then run `mkdir car-seg` to create a directory for our project. Run `cd car-seg` to go into this folder

Then we will create a virtual environment in this directory for the project. For this run the following command\
`python3 -m venv .venv` (make sure you are in the directory of car-seg as it will be easier to have the venv in the same directory as the project)

Now activate the virtual environment by running `source .venv/bin/activate` (still inside the car-seg directory). You should see (.venv) in front of your command line prompt, this means the virtual environment is active. \
You can deactivate by running `deactivate` and activate back using `source .venv/bin/activate` again.

Then run the following to install pytorch while venv is active \
`pip3 install torch torchvision torchaudio`

Any other package can also be installed the same way while the venv is active



### WORKING AFTER INITIAL SETUP
First run the following command on your command line using your student id \
`ssh <student_id>@login.hpc.dtu.dk `

Then load modules by running the following commands\
`module load python3/3.10.12`\
`module load cuda/12.1`

Go into the project dir\
`cd Desktop/car-seg`

Run the following to activate venv \
`source .venv/bin/activate`

Start working\
