# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:07:54 2018

@author: prajw
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

deliveries = pd.read_csv('deliveries.csv')
matches = pd.read_csv('matches.csv')

#Let us inspect the two Dataframes
deliveries.head()
deliveries.tail()

#Deliveries DF has 150460 rows and 21 columns
# Each row in the Deliveries DF represents 1 delivery. So the data is about 150460 deliveries in IPL.

matches.head()
matches.tail()

#Matches DF has 636 rows and 18 columns.
#Each row in the matches DF represents 1 match. So the data is about all 636 games played in IPL.

#Let us drop umpire3 column from match DF as it has NaN for all.
matches.drop('umpire3',axis = 1, inplace = True)

#Inspecting the data further we see that player_dismissed, dismissal_kind and fielder have lot of Null values
deliveries.info()

#let us take a look at player_dismissed
deliveries['player_dismissed'].head()

#Looking at player_dismissed column we can see that NaN does not suggest missing value but only that
#no player was dismissed.
#Let us fill in the NaNs with 'No Player dismissed'
deliveries['player_dismissed'].fillna('No Player dismissed', inplace = True)

#Similarly for dismissal_kind
deliveries['dismissal_kind'].fillna('No Player dismissed', inplace = True)

deliveries['fielder']
deliveries['fielder'].fillna('No dismissal/Fielder not involved', inplace = True)

#Inspect the data again
deliveries.info()
#No NA values at all

#Few columns in matches have null values but not a lot
matches.info()

#We can see that the city has NaN values only for season 2014. Not sure why.
matches[matches['city'].isnull()]

#We can see that the winner column has NaN's whenever there was no result probably due to weather.
matches[matches['winner'].isnull()]

#Let us change NaN's to match abondoned due to weather.
matches['winner'].fillna('No winner-Match abondonded due to weather', inplace = True)
matches['player_of_match'].fillna('No player of match-Match abondonded due to weather', inplace = True)

#Umpire 1 and 2 is not unknown for a single game in season 2017. Let us leave as it is for now.
matches[matches['umpire1'].isnull()]


#Chris Gayle has the maximum Player of Match awards.
#the code used is very basic but gets the job done easily
ax = matches['player_of_match'].value_counts().head(10).plot.bar(width=.8, color='R')  #counts the values corresponding 
# to each batsman and then filters out the top 10 batsman and then plots a bargraph 
ax.set_xlabel('player_of_match') 
ax.set_ylabel('count')
#for p in ax.patches:
#    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+0.25))
plt.show()


#
matches['team1'].unique()

#Let us replace the teams with abbreviations
matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW','RPS'],inplace=True)

deliveries.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW','RPS'],inplace=True)


print("Total matches played: ", matches.id.nunique())
print("Venues played at: ", matches.city.unique())
print("\n Team :", matches.team1.unique())

print("Total venues played at: ", matches.city.nunique())
print("Total umpires: ", matches.umpire1.nunique())

print((matches['player_of_match'].value_counts()).idxmax(),' : has most man of the match awards')
print((matches['winner'].value_counts()).idxmax(),' : has won most number of matches')

#Let us find the match with highest margin of victory by runs
df = matches.iloc[[matches['win_by_runs'].idxmax()]]
df[['season', 'city', 'team1', 'team2', 'winner', 'win_by_runs']]

#Mumbai Indians(MI) defeated Delhi Daredevils(DD) with the highest run difference

df = matches.iloc[[matches['win_by_wickets'].idxmax()]]
df[['season', 'city', 'team1', 'team2', 'winner', 'win_by_wickets']]

#Kolkata Knight Riders(KKR) defeated Gujrat Lions(GL) with the highest wins by wickets

print('Toss Decisions in %\n',((matches['toss_decision']).value_counts())/577*100)

#Toss decision across seasons
plt.subplots(figsize=(10,6))
sns.countplot(x='season',hue='toss_decision',data=matches)
plt.show()

#Maximum toss winners
ax = matches['toss_winner'].value_counts().plot.bar(width = 0.8)
plt.show()      

#Is Toss Winner Also the Match Winner?

df = matches[matches['winner']  == matches['toss_winner']]
slices = [len(df), 577 - len(df)]
labels = ["Yes", "No"]
plt.pie(slices, labels = labels,startangle=90,shadow=True,explode=(0,0),autopct='%1.1f%%',colors=['r','g'])
fig = plt.gcf()
fig.set_size_inches(6,6)
plt.show()

#Toss winner is not necessarily the match winner.

#Matches played across each season
matches['season'].value_counts()

sns.countplot(x = 'season', data = matches, palette = "Set1")
plt.show()

#Runs across seasons

batsmen = matches[['id', 'season']].merge(deliveries, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)
season = batsmen.groupby(['season'])['total_runs'].sum().reset_index()
season['total_runs'].plot(marker = 'o')
plt.show()

#There was a decline in total runs from 2008 to 2009.But there after there was a substantial 
#increase in runs in every season until 2013, but from next season there was a slump in the 
#total runs. But the number of matches are not equal in all seasons. We should check the 
#average runs per match in each season


#Average runs per match in each Season
plt.subplots(figsize=(10,6))
avgruns_each_season = matches.groupby(['season']).count().id.reset_index()
avgruns_each_season.rename(columns={'id':'matches'},inplace=1)
avgruns_each_season['total_runs']=season['total_runs']
avgruns_each_season['average_runs_per_match']=avgruns_each_season['total_runs']/avgruns_each_season['matches']
avgruns_each_season['average_runs_per_match'].plot(marker='o')
plt.show()

#Sixes and Fours Across the Season
season_6s = batsmen.groupby(['season'])['batsman_runs'].agg(lambda x: (x == 6).sum()).reset_index()
season_4s = batsmen.groupby(['season'])['batsman_runs'].agg(lambda x: (x == 4).sum()).reset_index()
season_boundaries = season_6s.merge(season_4s, left_on = 'season', right_on = 'season', how = 'left')
season_boundaries=season_boundaries.rename(columns={'batsman_runs_x':'6"s','batsman_runs_y':'4"s'})
season_boundaries[['6"s','4"s']].plot(marker = 'o')
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()

#Total Matches
matches_played_byteams=pd.concat([matches['team1'],matches['team2']])
matches_played_byteams=matches_played_byteams.value_counts().reset_index()
matches_played_byteams.columns=['Team','Total Matches']
matches_played_byteams['wins']=matches['winner'].value_counts().reset_index()['winner']
matches_played_byteams.set_index('Team',inplace=True)


#Runs Per Over By Teams Across Seasons
runs_per_over = deliveries.pivot_table(index=['over'],columns='batting_team',values='total_runs',aggfunc=sum)
runs_per_over[(matches_played_byteams[matches_played_byteams['Total Matches']>50].index)].plot(color=["b", "r", "#Ffb6b2", "g",'brown','y','#6666ff','black','#FFA500']) #plotting graphs for teams that have played more than 100 matches
x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
plt.xticks(x)
plt.ylabel('total runs scored')
fig=plt.gcf()
fig.set_size_inches(16,8)
plt.show()


#Favorite grounds
plt.subplots(figsize=(10,15))
ax = matches.venue.value_counts().sort_values(ascending  = True).plot.barh(width = .9)
ax.set_xlabel('Counts')
ax.set_ylabel('Grounds')
plt.show()

#Each season winner
for i in range(2008, 2017):
    df = (matches[matches['season'] == i]).iloc[-1]
    print(df[[1,10]])

#Super overs 
print("\n Total matches with super over: ", deliveries[deliveries['is_super_over'] == 1].match_id.nunique())

#Teams who have not played super over
teams=['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW']
play = deliveries[deliveries['is_super_over'] == 1].batting_team.unique()
play = list(play)

print("The teams which have not played super-over are: ", list(set(teams) - set(play)))


#Favorite Umpires
plt.subplots(figsize = (10, 6))
umpires = pd.concat([matches['umpire1'], matches['umpire2']])
ax = umpires.value_counts().head(10).plot.bar(width = 0.8, color = 'R')
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x() + 0.15, p.get_height() + 0.25))
plt.show

#200 + scores

high_scores = deliveries.groupby(['match_id','inning','batting_team', 'bowling_team'])['total_runs'].sum().reset_index()
high_scores = high_scores[high_scores['total_runs'] >= 200]
high_scores.nlargest(10, 'total_runs')

fig, ax = plt.subplots(1,2)
sns.countplot(high_scores['batting_team'],ax=ax[0])
sns.countplot(high_scores['bowling_team'],ax=ax[1])
plt.xticks(rotation=90)
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.show()

print("The teams which have not scored 200+ runs: ", list(set(teams) - set(high_scores['batting_team'])))
print("The teams which have not conceeded 200+ runs: ", list(set(teams) - set(high_scores['bowling_team'])))

#
high=deliveries.groupby(['match_id', 'inning','batting_team','bowling_team'])['total_runs'].sum().reset_index()
high.set_index(['match_id'],inplace=True)
high['total_runs'].max()
high.columns
high=high.rename(columns={'total_runs':'count'})
high=high[high['count']>=200].groupby(['inning','batting_team','bowling_team']).count()
high

#
high_scores = deliveries.groupby(['match_id', 'inning', 'batting_team', 'bowling_team'])['total_runs'].sum().reset_index()
high_scores1 = high_scores[high_scores['inning'] == 1]
high_scores2 = high_scores[high_scores['inning'] == 2]

high_scores1 = high_scores1.merge(high_scores2[['match_id', 'inning', 'total_runs']], on = 'match_id')
high_scores1.rename(columns={'inning_x':'inning_1','inning_y':'inning_2','total_runs_x':'inning1_runs','total_runs_y':'inning2_runs'}, inplace = True)

high_scores1 = high_scores1[high_scores1['inning1_runs'] >=200]
high_scores1['is_score_chased'] = 1
high_scores1['is_score_chased'] = np.where(high_scores1['inning1_runs'] <= high_scores1['inning2_runs'], 'yes', 'no')

#Visualising that
slices = high_scores1['is_score_chased'].value_counts().reset_index().is_score_chased
list(slices)
labels = ['target_not_chased', 'target_chased']
plt.pie(slices, labels = labels, colors=['#1f2ff3', '#0fff00'],startangle=90,shadow=True,explode=(0,0.1),autopct='%1.1f%%')
fig = plt.gcf()
fig.set_size_inches(6,6)
plt.show()

#It seems to be clear that team batting first and scoring 200+ runs, has a very high probablity of winning the match.

#Top 10 batsman in terms of runs
plt.subplots(figsize = (10, 6))
max_runs=deliveries.groupby(['batsman'])['batsman_runs'].sum()
ax=max_runs.sort_values(ascending=False)[:10].plot.bar(width=0.8,color='R')
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x() + 0.1), p.get_height() + 1, fontsize = 13)
plt.show()


#Bowlers with most wickets
dismissal_kinds = ["bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]  #since run-out is not creditted to the bowler
ct = deliveries[deliveries["dismissal_kind"].isin(dismissal_kinds)]
ct['bowler'].value_counts()[:10].plot.bar(width = 0.8, color = 'B')
plt.show()

#Fielders with most catches
ct1 = deliveries[deliveries["dismissal_kind"] == 'caught']
ax = ct1['fielder'].value_counts()[:10].plot.bar(width = 0.8, color = 'R')
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x() + 0.125, p.get_height() + 1))
plt.show()

#Maximum overs
overs = deliveries.groupby(['bowler']).sum()
overs['total_balls'] = deliveries['bowler'].value_counts()
overs['overs'] = (overs['total_balls']//6)

overs[overs['overs']>200].sort_values(by = 'overs', ascending = False)['overs'].head(5).reset_index()

#Economy
overs['economy'] = overs['total_runs']/overs['overs']
overs[overs['overs']>300].sort_values(by = 'economy')[:10].economy.reset_index().T

#Purple Caps Each Season (Maximum Wickets By Bowler per Season)
dismissal_kinds = ["bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]  #since run-out is not creditted to the bowler
purple=deliveries[deliveries["dismissal_kind"].isin(dismissal_kinds)]
purple=purple.merge(matches,left_on='match_id',right_on='id',how='outer')
purple=purple.groupby(['season','bowler'])['dismissal_kind'].count().reset_index()
purple=purple.sort_values('dismissal_kind',ascending=False)
purple=purple.drop_duplicates('season',keep='first').sort_values(by='season')
purple.columns=[['season','bowler','count_wickets']]

#Top Individual Scores
top_scores = deliveries.groupby(['match_id','batsman', 'batting_team'])['batsman_runs'].sum().reset_index()
top_scores.sort_values('batsman_runs', ascending = 0).head(10)
top_scores.nlargest(10, 'batsman_runs')

#Individual Scores By Top Batsman each Inning
swarm=['CH Gayle','V Kohli','G Gambhir','SK Raina','YK Pathan','MS Dhoni','AB de Villiers','DA Warner']
scores = deliveries.groupby(['match_id', 'batsman', 'batting_team'])['batsman_runs'].sum().reset_index()
scores = scores[top_scores['batsman'].isin(swarm)]
sns.swarmplot(x = 'batsman' , y = 'batsman_runs', data = scores, hue = 'batting_team', palette = 'Set1')
fig=plt.gcf()
fig.set_size_inches(14,8)
plt.ylim(-10,200)
plt.show()

#
#Observations:
#Chris Gayle has the highest Individual Score of 175 and Highest Number of Centuries i.e 5
#MS Dhoni and Gautam Gambhir have never scored a Century.
#V Kohli has played only for 1 IPL Team in all seasons i.e RCB

#Runs Scored By Batsman Across Seasons

a = batsmen.groupby(['season', 'batsman'])['batsman_runs'].sum().reset_index()
a = a.groupby(['season','batsman'])['batsman_runs'].sum().unstack().T
a['Total'] = a.sum(axis = 1)
a = a.sort_values(by = 'Total', ascending = 0)[:5]
a.drop('Total',axis=1,inplace=True)
a.T.plot(color=['red','blue','#772272','green','#f0ff00'],marker='o')
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()


#How do the top batsmen score?
a=batsmen.groupby(['batsman','batsman_runs'])['total_runs'].count().reset_index()
b=max_runs.sort_values(ascending=False)[:10].reset_index()
c=b.merge(a,left_on='batsman',right_on='batsman',how='left')
c.drop('batsman_runs_x',axis=1,inplace=True)
c.set_index('batsman',inplace=True)
c.columns=['type','count']
c=c[(c['type']==1)|(c['type']==2)|(c['type']==4)|(c['type']==6)]
cols=['type','count']
c.reset_index(inplace=True)
c=c.pivot('batsman','type','count')

#Who Has Taken Most Wickets Of The Top Batsman
gayle = deliveries[deliveries['batsman'] == 'CH Gayle']
gayle = gayle[gayle['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]
gayle = gayle.groupby(['bowler']).count().sort_values(by = 'dismissal_kind', ascending = 0).dismissal_kind[:1].reset_index()
gayle['batsman'] = 'CH Gayle'

kohli = deliveries[deliveries['batsman'] == 'V Kohli']
kohli = kohli[kohli['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]
kohli = kohli.groupby(['bowler']).count().sort_values(by = 'dismissal_kind', ascending = 0).dismissal_kind[:1].reset_index()
kohli['batsman'] = 'V Kohli'

raina=deliveries[deliveries['batsman']=='SK Raina']
raina=raina[raina['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]
raina=raina.groupby('bowler').count().sort_values(by='dismissal_kind',ascending=0).dismissal_kind[:1].reset_index()
raina['batsman']='SK Raina'

abd=deliveries[deliveries['batsman']=='AB de Villiers']
abd=abd[abd['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]
abd=abd.groupby('bowler').count().sort_values(by='dismissal_kind',ascending=0).dismissal_kind[:1].reset_index()
abd['batsman']='AB de Villiers'

msd=deliveries[deliveries['batsman']=='MS Dhoni']
msd=msd[msd['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]
msd=msd.groupby('bowler').count().sort_values(by='dismissal_kind',ascending=0).dismissal_kind[:1].reset_index()
msd['batsman']='MS Dhoni'


gg=deliveries[deliveries['batsman']=='G Gambhir']
gg=gg[gg['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]
gg=gg.groupby('bowler').count().sort_values(by='dismissal_kind',ascending=0).dismissal_kind[:1].reset_index()
gg['batsman']='G Gambhir'

rohit=deliveries[deliveries['batsman']=='RG Sharma']
rohit=rohit[rohit['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]
rohit=rohit.groupby('bowler').count().sort_values(by='dismissal_kind',ascending=0).dismissal_kind[:1].reset_index()
rohit['batsman']='RG Sharma'

uthapa=deliveries[deliveries['batsman']=='RV Uthappa']
uthapa=uthapa[uthapa['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]
uthapa=uthapa.groupby('bowler').count().sort_values(by='dismissal_kind',ascending=0).dismissal_kind[:1].reset_index()
uthapa['batsman']='RV Uthappa'

dhawan=deliveries[deliveries['batsman']=='S Dhawan']
dhawan=dhawan[dhawan['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]
dhawan=dhawan.groupby('bowler').count().sort_values(by='dismissal_kind',ascending=0).dismissal_kind[:1].reset_index()
dhawan['batsman']='S Dhawan'

warn=deliveries[deliveries['batsman']=='DA Warner']
warn=warn[warn['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]
warn=warn.groupby('bowler').count().sort_values(by='dismissal_kind',ascending=0).dismissal_kind[:1].reset_index()
warn['batsman']='DA Warner'

new = gayle.append([kohli,raina,abd,msd,gg,rohit,uthapa,dhawan,warn])
new = new[['batsman','bowler','dismissal_kind']]
new.columns=['batsman','bowler','No_of_Dismissals']
new


#Orange Caps Each Season(Highest Run Getter per Season)
orange = matches[['id', 'season']]
orange = orange.merge(deliveries, left_on = 'id', right_on = 'match_id', how = 'left')
orange = orange.groupby(['season', 'batsman'])['batsman_runs'].sum().reset_index()
orange = orange.sort_values(by = 'batsman_runs', ascending = 0)
orange = orange.drop_duplicates(subset=["season"],keep="first")
orange = orange.sort_values(by = 'season')

plt.subplots(figsize=(10,6))
sns.barplot(x = 'season', y = 'batsman_runs', hue = 'batsman', data = orange)
plt.show()

#Top 20 Bowlers
bowler = deliveries.groupby('bowler').sum().reset_index()
bowl = deliveries['bowler'].value_counts().reset_index()
bowler = bowler.merge(bowl, left_on = 'bowler', right_on = 'index', how = 'left')
bowler = bowler[['bowler_x','total_runs','bowler_y']]
bowler.columns=[['bowler','runs_given','balls']]
bowler['overs'] = (bowler['balls']//6)

dismissal_kinds = ["bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]  
ct=deliveries[deliveries["dismissal_kind"].isin(dismissal_kinds)]
ct = ct['bowler'].value_counts()[:20].reset_index()

bowler = bowler.merge(ct, left_on = 'bowler', right_on = 'index', how = 'left').dropna()
bowler =bowler[['bowler_x','runs_given','overs','bowler_y']]
bowler.columns=[['bowler','runs_given','overs','wickets']]
bowler['economy'] = (bowler['runs_given']/bowler['overs'])
bowler.head()

x = ['category_admin','category_beauty', 'category_education', 'category_general', 'category_manufacturing', 'category_medical','category_other', 'category_restaurant', 'category_retail', 'category_sales', 'category_transportation']
y = [21, 12, 8, 28, 6, 17, 47, 110, 94, 57, 18]

sns.barplot(x, y)
plt.ylabel('No of Applications for each job category')
plt.xlabel('Job Category')
plt.title('Distribution of Applications across jobs')
plt.xticks(rotation=90)
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.show()



#Extras And Wickets
extras=['wide_runs','bye_runs','legbye_runs','noball_runs']
sizes=[5161,680,3056,612]

dismiss=["run out","bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]
ct=deliveries[deliveries["dismissal_kind"].isin(dismiss)]
bx=ct.dismissal_kind.value_counts()[:10]
bx

#Teams with maximum Boundaries 
ax = deliveries[deliveries['batsman_runs'] == 6].batting_team.value_counts().reset_index()
ax2 = deliveries[deliveries['batsman_runs'] == 4].batting_team.value_counts().reset_index()

ax = ax.merge(ax2, left_on = 'index', right_on = 'index', how = 'left')
ax.columns = [['team', "6's", "4's"]]
ax


#How to win Finals??
finals = matches.drop_duplicates(subset = 'season', keep = 'last')
finals=finals[['id','season','city','team1','team2','toss_winner','toss_decision','winner']]
most_finals = pd.concat((finals['team1'], finals['team2'])).value_counts().reset_index()
most_finals.columns = [['team', 'count']]

xyz = finals['winner'].value_counts().reset_index()
most_finals = most_finals.merge(xyz, left_on = 'team', right_on = 'index', how = 'outer')
most_finals.drop(['index'], axis = 1, inplace = True)
most_finals = most_finals.replace(np.NaN, 0)

most_finals.set_index('team', inplace = True)
most_finals.columns = [['finals_played', 'won_count']]

most_finals.plot.bar(width = 0.8)
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()

#Do toss winners win finals
df = finals[finals['toss_winner'] == finals['winner']]
slices = [len(df), 9 - len(df)]
labels = ["Yes", "No"]
plt.pie(slices, labels = labels,startangle=90,shadow=True,explode=(0,0.1),autopct='%1.1f%%',colors=['r','g'])
fig = plt.gcf()
fig.set_size_inches(6,6)
plt.show()

#Batting Or Fielding For Toss Winners
finals['is_tosswin_matchwin']=finals['toss_winner']==finals['winner']
sns.countplot(x='toss_decision',hue='is_tosswin_matchwin',data=finals)
plt.show()

#Total Matches vs Wins for Teams 
matches_played_by_teams = pd.concat([matches['team1'], matches['team2']])
matches_played_by_teams = matches_played_by_teams.value_counts().reset_index()
matches_played_by_teams.columns = [['Teams','Played']]
matches_played_by_teams['wins'] = matches['winner'].value_counts().reset_index()['winner']
matches_played_by_teams.set_index('Teams', inplace = True)


#Runs Per Over By Teams Across Seasons
runs_per_over = deliveries.pivot_table(index = ['over'], columns = 'batting_team', values = 'total_runs', aggfunc = sum)
runs_per_over[(matches_played_by_teams[matches_played_by_teams['Played']>50].index)].plot(color=["b", "r", "#Ffb6b2", "g",'brown','y','#6666ff','black','#FFA500']) #plotting graphs for teams that have played more than 100 matches
x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
plt.xticks(x)
plt.ylabel('total runs scored')
fig=plt.gcf()
fig.set_size_inches(16,8)
plt.show()

#Team1 vs Team2

def team1_vs_team2(team1, team2):
    mt1=matches[((matches['team1']==team1)|(matches['team2']==team1))&((matches['team1']==team2)|(matches['team2']==team2))]
    sns.countplot(x = 'season', hue = 'winner', data = mt1, palette= 'Set3')
    plt.xticks(rotation = 'vertical')
    leg = plt.legend(loc = 'upper center')
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    plt.show()

team1_vs_team2('KKR', 'MI')

#Matches Won By A Team Against Other Teams
def comparator(team1):
    teams = ['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW']
    teams.remove(team1)
    opponents = teams.copy()
    mt1 = matches[((matches['team1'] == team1)|(matches['team2'] == team1))]
    for i in opponents:
        mask = (((mt1['team1']==i)|(mt1['team2']==i)))&((mt1['team1']==team1)|(mt1['team2']==team1))
        mt2 = mt1.loc[mask, 'winner'].value_counts().to_frame().T
        print(mt2)
        
comparator('MI')

#Top Individual Scores
top_scores = deliveries.groupby(['match_id','batsman'])['batsman_runs'].sum().reset_index()
top_scores = top_scores.sort_values(by = 'batsman_runs', ascending = 0)
top_scores.nlargest(10, 'batsman_runs')

#Purple Caps Each Season (Maximum Wickets By Bowler per Season)
dismissal_kinds
purple = deliveries[deliveries['dismissal_kind'].isin(dismissal_kinds)]
purple = purple.merge(matches, left_on = 'match_id', right_on = 'id', how = 'left')
purple = purple.groupby(['season','bowler'])['dismissal_kind'].count().reset_index()
purple = purple.sort_values(by = 'dismissal_kind', ascending = False)
purple = purple.drop_duplicates('season' , keep = 'first').sort_values(by = 'season', ascending = True)
purple.columns = [['Season', 'Bowler', 'No Of Wickets']]


#Orange Caps Each Season(Highest Run Getter per Season)