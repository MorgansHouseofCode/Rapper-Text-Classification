# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 19:17:18 2020

@author: Morgan
"""

#I am creating a method to grab all the albums that kendrick Lamar has and then grabbing all the info and create a dataframe that stores the info.
#Ideally I will have two text files: 1 a file of all the artist I want to use 2 a file that has the artist name as title and all the artist's albums in it


import pandas as pd
import os 
from  ScrapeAlbums import Genius
import time
client_access_token = #Its a secret
api = Genius(client_access_token) # Need this 
# =============================================================================

#creates columns from the json file so the dataframe can be filled out
def Createthecolumns(TempRapperDataFrame):
   TempRapperDataFrame['title'] = TempRapperDataFrame['songs'][0]['title']
   TempRapperDataFrame['Date Released'] = TempRapperDataFrame['songs'][0]['year']
   TempRapperDataFrame['album'] = TempRapperDataFrame['songs'][0]['album']
   TempRapperDataFrame['lyrics'] = TempRapperDataFrame['songs'][0]['lyrics']




#filles out the dataframe that will be appended to a lsit of data frames and merged into one dataframe
def JoinAlbums(Rapper):
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".json") and filename.startswith("Lyrics_{}".format(Rapper.replace(" ", ""))):
            print(filename)
            dfr = pd.read_json(filename)
    print(filename)
    TempRapperDataFrame = pd.DataFrame(data= dfr)
    Createthecolumns(TempRapperDataFrame)
         #Section to grab the important parts 
#         #see If I can implement the lambda functions I have below in the data cleaning section to clean this code section up
    n = 0
    while n < len(TempRapperDataFrame):
        
        TempRapperDataFrame['title'][n] = TempRapperDataFrame['songs'][n]['title'] #Grabs the title
        TempRapperDataFrame['Date Released'][n] = TempRapperDataFrame['songs'][n]['year'] # grabs the date realsed 
        TempRapperDataFrame['album'][n] = TempRapperDataFrame['songs'][n]['album'] #the album the song is in
        TempRapperDataFrame['lyrics'][n] = TempRapperDataFrame['songs'][n]['lyrics'] # the TempRapperDataFrame of the song
        n+=1
    return TempRapperDataFrame





#grabs rappers from the rappers text file
def GetRappersFromList(): #shit works
     for filename in os.listdir(os.getcwd()):
        if filename.endswith(".txt") and filename.startswith("Rapp"):
            with open(filename,"r+") as file:
                Artistnames = [(n) for n in file.read().splitlines() ]
                return(GetAlbumsFromFile(Artistnames))
    
#grabs the ablums from the text file
def GetAlbumsFromFile(Artistnames): 
    dict_of_rapper_album = {}
    for Rapper in Artistnames:
        for filename in os.listdir(os.getcwd()):
           # print(Rapper)
            if filename.endswith(".txt") and filename.startswith(Rapper):
                with open(filename,"r+") as file:
                    album = ([(n) for n in file.read().splitlines()])
                
        dict_of_rapper_album[Rapper] = album
    return(ScrapeThealbums(dict_of_rapper_album))


#Calls genuis API Script to scrape whole albums at a time 
def ScrapeThealbums(dict_of_rapper_album):
    list_of_dfs = []
    #count = 0
    #this is how you will grab the parts you need 
    for Rapper in dict_of_rapper_album.keys():
        for Album in dict_of_rapper_album[Rapper]:
            while True:
                try:
                     Genius.search_album(api,Rapper,Album)
                     df=JoinAlbums(Rapper)
                     list_of_dfs.append(df)
                     break 
                except:
                    print("I timed out")
                    time.sleep(30)
                    pass
                    
           
    NewRapperDataFrame = pd.concat(list_of_dfs, axis=0)
    return NewRapperDataFrame

    

AdvanceRapperDataFrame=(GetRappersFromList())
# =============================================================================
#  Some Quick Data Cleaning
AdvanceRapperDataFrame=AdvanceRapperDataFrame.reset_index(drop=True)
AdvanceRapperDataFrame=AdvanceRapperDataFrame.drop(columns=['songs'])
#time.sleep(30)
DataFrameToPlayWith = AdvanceRapperDataFrame.copy() # to make a copy in case I mess up cleaning 
    
for i in AdvanceRapperDataFrame.columns:  # sees if there is a lsit and if there is it makes a string sepreated by ,
    if  isinstance(AdvanceRapperDataFrame[i][0],list):
        AdvanceRapperDataFrame[i] = (AdvanceRapperDataFrame[i].apply(','.join))

#create datetime columns and fixing the format to it
AdvanceRapperDataFrame['Date Released'] = pd.DatetimeIndex(AdvanceRapperDataFrame['Date Released'])
AdvanceRapperDataFrame['Year'] = AdvanceRapperDataFrame.apply(lambda row: row['Date Released'].year, axis = 1) # gets the year for every row and creates a column 
AdvanceRapperDataFrame['Month'] = AdvanceRapperDataFrame.apply(lambda row: row['Date Released'].month, axis = 1) # gets the year for every row and creates a column 
AdvanceRapperDataFrame['Day'] = AdvanceRapperDataFrame.apply(lambda row: row['Date Released'].day, axis = 1) # gets the year for every row and creates a column 

AdvanceRapperDataFrame.to_csv("RapperDataFrame.csv", index=False)

