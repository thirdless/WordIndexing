from mpi4py import MPI
import os
import shutil
from datetime import datetime
import json
import re
import sys

LOG_NAME = "last_run.log"
TEMP_NAME = "map"
TAG = 55

def log(text):
    text = "[Rank " + str(my_rank) + "][" + datetime.now().strftime("%H:%M:%S.%f") + "]: " + str(text) + "\n"

    # f = open(LOG_NAME, "a")
    # f.write(text)
    # f.close()

    print(text, end="")

def exitAll():
    for i in range(1, world_size):
        comm.send("exit", dest=i, tag=TAG)
    exit()

def masterProcess():
    FILES_NAME = ARG_FILES
    INDEX_NAME = ARG_INDEX

    #remove temporary, log and old index files
    if os.path.exists(LOG_NAME):
        os.remove(LOG_NAME)
    if os.path.exists(INDEX_NAME):
        shutil.rmtree(INDEX_NAME)
    if os.path.exists(TEMP_NAME):
        shutil.rmtree(TEMP_NAME)

    #get all files
    if not os.path.isdir(FILES_NAME):
        log("Missing the '" + FILES_NAME + "' directory, exitting.")
        exitAll()

    files = os.listdir(FILES_NAME)
    files_done = len(files)

    #get all the index characters
    letters = [chr(i) for i in range(97, 123)]
    numbers = [str(i) for i in range(0, 10)]
    characters = letters + numbers
    characters_done = len(characters)

    #continuously listening for messages from the workers
    while True:
        command = comm.recv(source=MPI.ANY_SOURCE, tag=TAG)

        #worker reported an error, closing the program
        if command == "abort":
            exitAll()
        #worker requests a file for the mapping step
        elif command.find("map") == 0:
            #worker id
            id = int(command.replace("map ", ""))
            #send a file if there are any left
            if len(files):
                comm.send("map " + files.pop(0), dest=id, tag=TAG)
        #worker finished mapping the file
        elif command.find("end_map") == 0:
            files_done -= 1
            #if there are no more files to be mapped, announce the workers they can start requesting for the reducing step
            if not files_done:
                for i in range(1, world_size):
                    comm.send("request_reduce", dest=i, tag=TAG)
        # worker requests a character to start indexing in the reducing step
        elif command.find("reduce") == 0:
            #worker id
            id = int(command.replace("reduce ", ""))
            #send a character if there are any left
            if len(characters):
                comm.send("reduce " + characters.pop(0), dest=id, tag=TAG)
        #worker finished reducing a character
        elif command.find("end_reduce") == 0:
            characters_done -= 1
            #program finished successfully, closing the program
            if not characters_done:
                shutil.rmtree(TEMP_NAME)
                exitAll()

def workerMap(command, mapCommand):
    FILES_NAME = ARG_FILES
    INDEX_NAME = ARG_INDEX

    index = {}
    #file name from master
    name = command.replace("map ", "")
    #opening the file in binary mode
    f = open(FILES_NAME + "/" + name, "rb")

    words = 0
    term = 0

    #reading the file line by line
    for line in f:
        #splitting the line into separate words
        words = line.split()
        #iterate through each word
        for i in range(0, len(words)):
            #try-catch for the decoding mechanism, skipping faulty words
            try:
                term = words[i].decode("utf-8").lower()
                #filtering the non-alphanumeric characters at the start and the end of the word,
                #keeping the special chars only inside of the word
                term = re.sub(r"^\W+|\W+$", "", term)
            except:
                continue

            #mapping only words that contain 3 or more characters
            if len(term) < 3:
                continue
            
            #incrementing the count of the word
            if term in index:
                index[term] += 1
            else:
                index[term] = 1

    f.close()

    #create temp directory if it doesn't exist
    if not os.path.exists(TEMP_NAME):
        os.mkdir(TEMP_NAME)

    #write json data in tmp file
    f = open(TEMP_NAME + "/" + name + ".tmp", "w")
    f.write(json.dumps(index))
    f.close()

    #after mapping a file, message the master and request a new one
    comm.send("end_map", dest=0, tag=TAG)
    comm.send(mapCommand, dest=0, tag=TAG)

def workerReduce(command, reduceCommand):
    FILES_NAME = ARG_FILES
    INDEX_NAME = ARG_INDEX

    #character from master
    match = command.replace("reduce ", "")
    index = {}

    #checking if the files from the mapping step exist
    if not os.path.exists(TEMP_NAME):
        log("No files to reduce, aborting")
        comm.send("abort", dest=0, tag=TAG)
        exit()

    #getting all the mapped files
    files = os.listdir(TEMP_NAME)
    terms = 0

    for i in range(0, len(files)):
        file_name = files[i]
        #original file name
        doc = file_name.rsplit(".tmp", 1)[0]

        #load json object from file
        f = open(TEMP_NAME + "/" + file_name, "r")
        terms = json.loads(f.read())
        f.close()

        #iterate through each term
        for term in terms:
            #check if the word starts with the indexing character
            if term.find(match) == 0:
                #adding the document name to the indexing object
                if term not in index:
                    index[term] = {}
                index[term][doc] = terms[term]
    
    #creating the index folder if it doesnt exist
    if not os.path.exists(INDEX_NAME):
        os.mkdir(INDEX_NAME)

    #writing the index json data to file in a pretty-print format
    f = open(INDEX_NAME + "/" + match + ".idx", "w")
    f.write(json.dumps(index, indent=4, sort_keys=True))
    f.close()

    #after reducing a character, message the master and request a new one
    comm.send("end_reduce", dest=0, tag=TAG)
    comm.send(reduceCommand, dest=0, tag=TAG)

def workerProcess():
    MAP_COMMAND = "map " + str(my_rank)
    REDUCE_COMMAND = "reduce " + str(my_rank)
    
    #start the mapping step by requesting a file
    comm.send(MAP_COMMAND, dest=0, tag=TAG)

    #continuously listening for commands from master
    while True:
        command = comm.recv(source=0, tag=TAG)

        #exit request
        if command == "exit":
            exit()
        #file mapping command
        elif command.find("map") == 0:
            workerMap(command, MAP_COMMAND)
        #start requesting for the reducing step
        elif command.find("request_reduce") == 0:
            comm.send(REDUCE_COMMAND, dest=0, tag=TAG)
        #character reducing command
        elif command.find("reduce") == 0:
            workerReduce(command, REDUCE_COMMAND)

def main():
    global my_rank, world_size, comm, ARG_FILES, ARG_INDEX

    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    my_rank = comm.Get_rank()

    if len(sys.argv) < 3:
        if my_rank == 0:
            log("Missing starting arguments:\n\t1 - files folder name\n\t2 - result folder name")
        return

    # FILES_NAME = sys.argv[1]
    # INDEX_NAME = sys.argv[2]

    ARG_FILES = sys.argv[1]
    ARG_INDEX = sys.argv[2]

    #master process
    if my_rank == 0:
        masterProcess()
    #worker processes
    else:
        workerProcess()

if __name__ == "__main__":
    main()
