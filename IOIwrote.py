def ReadInt(qstStr:str="",inMin:int=None,inMax:int=None)->int:
    valid:bool=False
    userInput:int

    while valid==False:
        try:
            userInput=int(input(qstStr))
            valid=True

            if inMin!=None:
                if userInput<inMin:
                    valid=False

            if inMax!=None:
                if userInput>inMax:
                    valid=False

        except ValueError:
            valid=False

        if valid==False:
            print("Invalid input!")

    return userInput

def ReadFloat(qstStr:str="",inMin:float=None,inMax:float=None)->float:
    valid:bool=False
    userInput:float

    while valid==False:
        try:
            userInput=float(input(qstStr))
            valid=True

            if inMin!=None:
                if userInput<inMin:
                    valid=False

            if inMax!=None:
                if userInput>inMax:
                    valid=False

        except ValueError:
            valid=False

        if valid==False:
            print("Invalid input!")

    return userInput

def ReadStr(qstStr:str="", validStr:list=None)->str:
    valid:bool=False
    userInput:str=""

    while valid==False:

        userInput=input(qstStr)

        if userInput in validStr:
            valid=True

        else:
            print("Invalid input!")

    return userInput

def YNDecision(decisionStr:str="")->bool:
    userInput:str=ReadStr(qstStr=decisionStr,validStr=['y','Y','n','N'])

    if userInput in ['y','Y']:
        return True

    else:
        return False

if __name__ == "__main__":

    exit:bool=False

    while exit==False:

        userInput = ReadInt("int Input: ")
        print("User's int inputed: ", userInput)
        userInput = ReadFloat("float Input: ")
        print("User's float inputed: ", userInput)
        exit=YNDecision(decisionStr="Exit? (Y/N)\n")

    input()