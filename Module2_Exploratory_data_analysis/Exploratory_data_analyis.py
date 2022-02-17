#%%
import pandas as pd
pd.options.display.max_columns = None

df = pd.read_csv("https://aicore-files.s3.amazonaws.com/Data-Science/multiple_choice_responses.csv")
df
# %%
## There's a file in the DATA folder called "questions_only.csv". Load in the dataset and all print the questions
q_df = pd.read_csv("https://aicore-files.s3.amazonaws.com/Data-Science/questions_only.csv")
for i, question in enumerate(q_df.iloc[0]):
    print(i, "\t", question)
    
# From this preview, here is what I noticed:
# There are a lot of questions (lots of data to analyse)
# Some of the questions allow for multiple inputs. For these questions, the header row/column names have _ appended to them, followed by some text.
# If the text is OTHER_TEXT then it seems to indicate that following a categorical question, a text field giving the option for the recipient to expand is provided. It looks like -1 is means that the user did not write anything.
# If the text is PART_N then it seems to be a checkbox question (i.e. tick all that apply)
# They are not mutually exclusive
# %%
# From this list, I'll extract these questions: Q: 1, 2, 3, 4, 5, 9, 10, 15, 18, 24, 28.

idx_to_keep = [1,2,3,4,5,9,10,15,18,24,28]

def extract_columns(df, idx_to_keep):
    
    new_df = pd.DataFrame() # empty dataframe
    df_col_list = df.columns.tolist()
    
    for i in idx_to_keep:
        column_name_base = "Q{}".format(i)
        column_index = [df_col_list.index(col_name) for col_name in df_col_list if col_name.startswith(column_name_base)][0]
               
        next_column_name_base = "Q{}".format(i+1)
        next_column_index = [df_col_list.index(col_name) for col_name in df_col_list if col_name.startswith(next_column_name_base)][0]
         
        col_idxs_to_extract = range(column_index, next_column_index)
        relevant_cols_df = df.iloc[:, col_idxs_to_extract]
        
        new_df = pd.concat([new_df, relevant_cols_df], axis=1)
        
    return new_df


df_orig = df.copy(deep=True)
df = extract_columns(df_orig, idx_to_keep)
df = df[1:]
df

# %%
df["Q2"] = df["Q2"].astype("category")
set(df["Q2"])
# %%
import plotly.express as px
px.histogram(df, "Q2", labels={"value": "Gender"}, title="Counts of Gender")
# %%
set(df["Q3"])
# So here are the values I think need updating:

# Hong Kong (S.A.R.)
# Iran, Islamic Republic of...
# United Kingdom of Great Britain and Northern Ireland
# Viet Nam
# South Korea
# Also notice there's an "Other
#%%
print("Percentage of 'Other':", df["Q3"].value_counts()["Other"]/len(df) * 100)

values_to_update = {"Q3": 
                    {"Hong Kong (S.A.R.)": "Hong Kong",
                     "Iran, Islamic Republic of...": "Iran",
                     "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                     "South Korea": "Republic of Korea",
                     "Viet Nam": "Vietnam"}}

## Using the replace method, update the values in the relevant column
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
df.replace(values_to_update, inplace=True)
set(df["Q3"])
# %%
import pycountry

## Create a new dataframe which will hold only the unique countries, their country codes and the number of instances of this country - WITHOUT "Other"
countries = df["Q3"][df["Q3"]!= "Other"].unique()
countries_df = pd.DataFrame(countries, columns=["Country"])
countries_df["Count"] = countries_df["Country"].map(df["Q3"].value_counts())

## Create a new column in the dataframe which has the ISO country codes
country_codes = []
for country in countries_df["Country"]:
    country_code = pycountry.countries.search_fuzzy(country)[0] # Take the first element returned from the search
    country_codes.append(country_code.alpha_3)

countries_df["Country Code"] = country_codes
countries_df
# %%
px.choropleth(countries_df, locations="Country Code", hover_name="Country", color="Count")
# %%
age_gender_df = df[["Q1", "Q2"]]
age_gender_groups = age_gender_df.groupby(["Q1", "Q2"]).size().unstack()
fig = px.bar(age_gender_groups, title="Count of Age per Gender", labels={"Q1": "Age", "value": "Count"})
fig.update_layout(legend_title_text='Gender')
# fig.update_layout(barmode="group")
fig.show()
# %%
# Education Analysis
# Produce two plots:
# The participants' formal education
# The count of formal education per gender. Display this is a grouped bar chart

#%%
fig = px.histogram(df, "Q4", height=800, title="Count of Education", labels={"value": "Education level"})
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()

edu_gender_df = df[["Q2", "Q4"]]
edu_gender_groups = edu_gender_df.groupby(["Q4", "Q2"]).size().unstack()
fig = px.bar(edu_gender_groups, title="Education level count per Gender",
             labels={"Q2": "Education", "value": "Count"},
             height=800)
fig.update_layout(legend_title_text='Gender', xaxis={'categoryorder':'total descending'})
# fig.update_layout(barmode="group")
fig.show()
# %%
# Let's create another diagram showing the same information but across 4 different plots

#%%
fig = px.histogram(df, "Q4", 
                   facet_col="Q2", 
                   color="Q2",
                   title="Counts of Education level per Gender",
                   labels={"Q4": "Education Level"},
                   height=1000, 
                   facet_col_wrap=2, 
                   facet_col_spacing=0.1,
                   )
fig.update_layout(showlegend=False, xaxis={'categoryorder':'total descending'})
fig.update_yaxes(matches=None, showticklabels=True)
# fig.update_xaxes(showticklabels=True)
fig.show()
# %%
# Sankey diagram
# We want five levels of education: Bachelor's, Master's, Doctoral, Professional, Other
## Create a new dataframe with just the education of surveyees where their education has been mapped to the above level
education_df = pd.DataFrame(df["Q4"])
education_df.rename(columns={"Q4": "Education Level"}, inplace=True)

values_to_update = {"Education Level": 
                    {"Some college/university study without earning a bachelor’s degree": "Other",
                     "No formal education past high school": "Other",
                     "I prefer not to answer": "Other"}}

education_df = education_df.replace(values_to_update)
set(education_df["Education Level"])
# %%
# Let's drop na's from Education Level
education_df.isna().sum()
education_df = education_df.dropna(subset=["Education Level"])
education_df.isna().sum()
# %%
## Add the gender, age and region columns to the new dataframe. Name the columns appropiately
cols_to_join = ["Q1", "Q2", "Q3"]
desired_col_names = ["Age", "Gender", "Region"]
for col, name in zip(cols_to_join, desired_col_names):
    education_df[name] = df[col]
    
education_df
# %%
# For visualisation purposes let's create:
# 1. wider age bins as 18-29, 30-49, 50-69 and 70+
# 2. group genders as "Male", "Female", "Other"
# 3. Convert countries to continents - apart from "India", "United States of America" and "Other"

## Overwrite the age and gender columns so that ages are now: 18-29, 30-49, 50-69 and 70+ and genders are "Male", "Female" and "Other"
values_to_update = {
    "Age": {"18-21": "18-29", "22-24": "18-29", "25-29": "18-29",
            "30-34": "30-49", "35-39": "30-49", "40-44": "30-49", "45-49": "30-49",
            "50-54": "50-69", "55-59": "50-69", "60-69": "50-69"
           },
    "Gender": {"Prefer not to say": "Other", "Prefer to self-describe": "Other"}
}

education_df = education_df.replace(values_to_update)
education_df
# %%
import pycountry_convert as pc
## Map countries to their relevant continents, unless the country is India, United States of America, or Other
countries_to_not_map = ["India", "United States of America", "Other"]
countries_to_map_to_continents = set(education_df["Region"])
for country in countries_to_not_map:
    countries_to_map_to_continents.discard(country)

countries_continent_dict = dict()
for country in countries_to_map_to_continents:
    country_alpha2 = pycountry.countries.search_fuzzy(country)[0].alpha_2
    continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    continent_name = pc.convert_continent_code_to_continent_name(continent_code)
    countries_continent_dict[country] = continent_name

to_update = {"Region": countries_continent_dict}
education_df = education_df.replace(to_update)
education_df
# %%
# Re-indexing the columns in the order we want them for the diagram because it'll be easier to work with
education_df = education_df.reindex(["Gender", "Age", "Region", "Education Level"], axis=1)

col_names = education_df.columns.tolist()
node_labels = []
num_categorical_vals_per_col = []
for col in col_names:
    uniques = education_df[col].unique().tolist()
    node_labels.extend(uniques)
    num_categorical_vals_per_col.append(len(uniques))
    
node_labels, num_categorical_vals_per_col
# %%
education_df.groupby(["Gender", "Age"]).size()["Female"]["18-29"]
# %%
import numpy as np
import random

source = []
target = []
value = []
colors = []
for i, num_categories in enumerate(num_categorical_vals_per_col):
    
    if i == len(num_categorical_vals_per_col)-1:
        break
    
    # index allows us to refer to the categories by index from the `node_labels` list
    start_index = sum(num_categorical_vals_per_col[:i])
    start_index_next = sum(num_categorical_vals_per_col[:i+1])
    end_index_next = sum(num_categorical_vals_per_col[:i+2])
#     print(start_index, start_index_next, end_index_next)
    
    # i can also give us the category column to refer to
    col_name = col_names[i]
    next_col_name = col_names[i+1]
    
    grouped_df = education_df.groupby([col_name, next_col_name]).size()
#     print(grouped_df)
    
    for source_i in range(start_index, start_index_next):
        for target_i in range(start_index_next, end_index_next):
            source.append(source_i)
            target.append(target_i)
            source_label = node_labels[source_i]
            target_label = node_labels[target_i]
            # if the index doesn't exist in the grouped_df, then the value is 0
            try:
                value.append(grouped_df[source_label][target_label])
            except:
                value.append(0)
            
            random_color = list(np.random.randint(256, size=3)) + [random.random()]
            random_color_string = ','.join(map(str, random_color))
            colors.append('rgba({})'.format(random_color_string))

print(source)
print(target)
print(value)

link = dict(source=source, target=target, value=value, color=colors)
# %%
import plotly.graph_objects as go

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = node_labels,
      color = "blue"
    ),
    link = link)])

fig.update_layout(title_text="Sankey Diagram (Gender, Age, Region, Education)", font_size=10)
fig.show()
#%%

# We'll sort the df by Age so our plot displays in the age order
df = df.sort_values(by=["Q1"])

fig = px.histogram(df, "Q4", facet_col="Q1",
             color="Q1",
             title="Counts of Education level per Age",
             labels={"Q1": "Age", "Q4": "Education Level"},
             height=1000, 
             facet_col_wrap=4, 
             facet_col_spacing=0.1)

fig.update_layout(showlegend=False, xaxis={'categoryorder':'total descending'})
fig.update_yaxes(matches=None, showticklabels=True)
# Interesting! The younger results are perhaps what we'd expect - 18-21 year olds typically aren't old enough to do masters degrees hence the number of bachelor's is higher for them. However, for almost every other age group, Master's degrees are prominent. Curiously, those over 70 are more likely to have a doctorate.

# %%
set(df["Q5"])
# %%
fig = px.histogram(df, "Q5", facet_col="Q1",
             color="Q1",
             title="Counts of Education level per Age",
             labels={"Q1": "Age", "Q5": "Job Role"},
             height=2000, 
             facet_col_wrap=2, 
             facet_col_spacing=0.1)

fig.update_layout(showlegend=False)
fig.update_xaxes(showticklabels=True, tickangle=45)
fig.update_yaxes(matches=None, showticklabels=True)
# %%
# Our data is in columns 18_p1, 18_p12
# Our first step will be to create a new column called "Known programming languages", and per row, create a comma separated list which contain the programming languages they know (obviously excluding NaNs)
programming_cols = ["Q18_Part_{}".format(str(i)) for i in range(1, 13)]
programming_df = df[programming_cols]
programming_df
# %%
programming_col = []
for row in programming_df.itertuples(index=False):
    languages_known = [language for language in row if isinstance(language, str)]
    programming_col.append(",".join(languages_known))
    
programming_df["languages_known"] = programming_col
programming_df
# %%
# Let's trim the new df so it only has our new col
programming_df.drop(labels=programming_cols, axis=1, inplace=True)
programming_df
# %%
# Assume blanks mean they don't know a language and replace both the blanks and "None" with "None/NA"
values_to_update = {"languages_known": {"": "None/NA", "None": "None/NA"}}
programming_df = programming_df.replace(values_to_update)
programming_df
# %%
# Let's use the get_dummies method to create new columns 
language_dummies = programming_df['languages_known'].str.get_dummies(sep=',')
language_dummies
# %%
fig = px.bar(language_dummies.sum(), labels={"index": "Programming Language"}, title="Count of Programming Languages")
fig.update_layout(showlegend=False, xaxis={'categoryorder':'total descending'})
fig.show()
# %%
# For 4. let's get the ages from the original dataframe and join them to this new dataframe
ages = df["Q1"]
language_dummies_with_age = language_dummies.join(ages).rename(columns={"Q1": "Age"})
language_dummies_with_age
# %%
programming_languages_by_age = language_dummies_with_age.groupby(["Age"]).sum()
px.bar(programming_languages_by_age)
# px.bar(programming_languages_by_age.T)
programming_languages_by_age = programming_languages_by_age.reindex(
    programming_languages_by_age.mean().sort_values().index, axis=1)
programming_languages_by_age = programming_languages_by_age.iloc[:, ::-1]
programming_languages_by_age
# %%
programming_languages_by_age_row_norm = programming_languages_by_age.div(programming_languages_by_age.sum(axis=1), axis=0)
# %%
from plotly.subplots import make_subplots

# programming_languages_by_age.index
programming_languages = programming_languages_by_age_row_norm.columns.tolist()
fig = make_subplots(4, 3, subplot_titles=programming_languages_by_age_row_norm.index)
for i, age_range in enumerate(programming_languages_by_age_row_norm.index):
    row = (i // 3) + 1
    col = (i % 3) + 1
    fig.add_trace(
        go.Bar(x=programming_languages, y=programming_languages_by_age_row_norm.iloc[i]),
        row=row, col=col
    )
fig.update_layout(showlegend=False, height=1000, title="Percent of Known Programming Languages by Age")
fig.update_yaxes(tickformat="%")
fig.show()
# # %%
# Looks like everyone likes Python! Younger people (who are most likely to be doing a bachelor's) have a relatively higher percentage of lower level languages like C, C++ and Java. This could be because they have to study these languages at university. As the data scientists specialise more in their career they seem to move away from these languages into more typical DSML related languages. The 60-69 group has a high level of R users relative to the other age groups, while it seems like most people over 70 don't know any programming languages.

# Perhaps more useful to us is what programming languages are popular against what job roles (Q5). Create a plot which demonstrates this
#%%
language_dummies_with_job = language_dummies.join(df["Q5"]).rename(columns={"Q5": "Job Title"})
language_dummies_with_job
# %%
## Group by Job Title, and aggregate the number, normalize each row, and sort the dataframe on the mean of the columns
languages_job_title_grouped = language_dummies_with_job.groupby(["Job Title"]).sum()
languages_job_title_grouped = languages_job_title_grouped.div(languages_job_title_grouped.sum(axis=1), axis=0)
languages_job_title_grouped = languages_job_title_grouped.reindex(
    languages_job_title_grouped.mean().sort_values().index, axis=1)
languages_job_title_grouped = languages_job_title_grouped.iloc[:, ::-1]
languages_job_title_grouped
# %%
px.imshow(languages_job_title_grouped, title="Heatmap of Programming Languages and Job Title")
# %%
## Plot the data!
programming_languages = languages_job_title_grouped.columns.tolist()
fig = make_subplots(4, 3, subplot_titles=languages_job_title_grouped.index)
for i, role in enumerate(languages_job_title_grouped.index):
    row = (i // 3) + 1
    col = (i % 3) + 1
#     numbers = langugages_job_title_grouped.iloc[i]
#     as_percent = [number / sum(numbers) for number in numbers]
    fig.add_trace(
        go.Bar(x=programming_languages, y=languages_job_title_grouped.iloc[i]),
        row=row, col=col
    )
fig.update_layout(showlegend=False, height=1000, title="Percent of Known Programming Languages Usage per Job Role")
fig.update_yaxes(tickformat="%")
fig.show()
# %%
# Findings:

# Python is averagly the most popular language amongst all job roles
# SQL is the most popular language amongst database engineers
# MATLAB is relatively more popular amongst research scientists than other job roles
# Statistician's prefer R over Python
# Student's and oftware engineers prefer C++ over R
# Let's do one more plot - a heatmap of the framework that each job role likes to use
#%%
## Plot the top 3 frameworks that each job role likes to use
# 28p1 - 28p12
## Create a dateframe which contains just the framework columns
framework_cols = ["Q28_Part_{}".format(str(i)) for i in range(1, 13)]
framework_df = df[framework_cols]
framework_df
# %%

Data Visualisation
Learning Objectives
Understand what EDA is, and why it is necesary
Practise applying various visualisation techniques
Learn what type of plot is appropiate for what situation
EDA stands for Exploratory Data Analysis and is a critical precursor to applying a model. As the name implies, it is all about exploring your data - validating that the dataset you'll be working on is clean, and without missing values. Perhaps most interestingly, however, is the ability to use various visualisation techniques on our data to gain an understanding of underlying trends between the variables provided.

I want to note that not all problems you will come across require the use of a model. Perhaps the task at hand is to "simply" provide visualisations and identify interesting facts which could not be done through a non-visual analysis. When we do want to use a model, it is important to have our hypothesis formulated. This is because the identification of what you're trying to find will be relevant in determining what parts of your data you explore. Either way, we'll be visualising data. To expand on why this is necessary, the image below demonstrates something known as Anscombe's Quartet:



(source)

Ascomebe's Quartet is case in point why visualising our data is of upmost importance. The image shows us that the summary statistics (e.g. mean, variance) for all the data is the same. However, as can be seen, the distributions which the data come from are wildly different. Had we not visualised our data, we would not have been able to trivially identify the relationships of the data.

Data cleaning and dealing with missing data falls under the EDA umbrella, and it will be a cyclical process where you explore your data, find out things wrong with it, clean it, and explore again.

We'll be working with the "multiple_choice_responses.csv" file from the 2019 Kaggle ML & DS Survey, which is a 35 question survey performed on Kaggle users regarding the state of data science and machine learning. From their abstract, this survey received 19,717 usable respondents from 171 countries and territories. If a country or territory received less than 50 respondents, we grouped them into a group named "Other" for anonymity. The task that we're going to assign ourself with this dataset is to identify what factors signifcantly impact the annual salary of those in DSML.

[ ]
## Load the dataset and return the first few rows
import pandas as pd
pd.options.display.max_columns = None

df = pd.read_csv("https://aicore-files.s3.amazonaws.com/Data-Science/multiple_choice_responses.csv")
df

[ ]
## There's a file in the DATA folder called "questions_only.csv". Load in the dataset and all print the questions
q_df = pd.read_csv("https://aicore-files.s3.amazonaws.com/Data-Science/questions_only.csv")
for i, question in enumerate(q_df.iloc[0]):
    print(i, "\t", question)
0 	 Duration (in seconds)
1 	 What is your age (# years)?
2 	 What is your gender? - Selected Choice
3 	 In which country do you currently reside?
4 	 What is the highest level of formal education that you have attained or plan to attain within the next 2 years?
5 	 Select the title most similar to your current role (or most recent title if retired): - Selected Choice
6 	 What is the size of the company where you are employed?
7 	 Approximately how many individuals are responsible for data science workloads at your place of business?
8 	 Does your current employer incorporate machine learning methods into their business?
9 	 Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice
10 	 What is your current yearly compensation (approximate $USD)?
11 	 Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?
12 	 Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice
13 	 On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice
14 	 What is the primary tool that you use at work or school to analyze data? (Include text response) - Selected Choice
15 	 How long have you been writing code to analyze data (at work or at school)?
16 	 Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice
17 	 Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice
18 	 What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice
19 	 What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice
20 	 What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice
21 	 Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice
22 	 Have you ever used a TPU (tensor processing unit)?
23 	 For how many years have you used machine learning methods?
24 	 Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice
25 	 Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice
26 	 Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice
27 	 Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice
28 	 Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice
29 	 Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice
30 	 Which specific cloud computing products do you use on a regular basis? (Select all that apply) - Selected Choice
31 	 Which specific big data / analytics products do you use on a regular basis? (Select all that apply) - Selected Choice
32 	 Which of the following machine learning products do you use on a regular basis? (Select all that apply) - Selected Choice
33 	 Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice
34 	 Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice
From this preview, here is what I noticed:

There are a lot of questions (lots of data to analyse)
Some of the questions allow for multiple inputs. For these questions, the header row/column names have _ appended to them, followed by some text.
If the text is OTHER_TEXT then it seems to indicate that following a categorical question, a text field giving the option for the recipient to expand is provided. It looks like -1 is means that the user did not write anything.
If the text is PART_N then it seems to be a checkbox question (i.e. tick all that apply)
They are not mutually exclusive
Analysing this data column by column is going to take way too long. So we're going to decide on factors we think may influence salary, and extract the relevant questions which meet this criteria from the list. This is partially why data science is considered an art - you may find yourself with a big dataset and find yourself unsure where to start analysing it. Using your hypothesis and identifying what you're trying to model, you need to use your best intuition as to the factors you think will be heavily influential to this. This is why domain expertise is important. But the more you explore your data with the initial concepts you had in mind, the more you'll end up learning about the wider dataset.

Salary (target)
Age
Gender
Residence
Education
Job role/Experience
Programming languages
ML frameworks
From this list, I'll extract these questions: Q: 1, 2, 3, 4, 5, 9, 10, 15, 18, 24, 28.

There are a couple others in there which would be relevant to analyse too - in an ideal world we would analyse them too, but time is limited here - and what is important is to teach you various visualiation techniques while building intuition as to what to look for in data.

Some of these questions encompass multiple columns in our dataframe. Extracting the relevant columns that we want isn't the most straightforward task. Take some time to try and implement something which returns a new dataframe which contains the relevant columns. If you're unsure how to proceed after a couple of minutes, click below to try and implement the method that I would use.

> Click here to find out how I would do this
Define a function which loops over a list of integers of questions we want to keep
For every iteration, work out the amount of columns from the current question to the next question in the dataframe (NOT the next question we want to extract)
Extract/concatenate from the current column position to the current column position + 'distance' (probably using the range() function from Python)
[ ]
  idx_to_keep = [1,2,3,4,5,9,10,15,18,24,28]

  def extract_columns(df, idx_to_keep):
      
      new_df = pd.DataFrame() # empty dataframe
      df_col_list = df.columns.tolist()
      
      for i in idx_to_keep:
          column_name_base = "Q{}".format(i)
          column_index = [df_col_list.index(col_name) for col_name in df_col_list if col_name.startswith(column_name_base)][0]
…

  df_orig = df.copy(deep=True)
  df = extract_columns(df_orig, idx_to_keep)
  df = df[1:]
  df

Still... a lot of data... well, we gotta start somewhere! Arbitrarily, let's start with Gender (Q2). We see that the data here is meant to be categorical, so after ensuring that's the case, let's simply plot the frequency of each of the values

[ ]
df["Q2"] = df["Q2"].astype("category")
set(df["Q2"])
{'Female', 'Male', 'Prefer not to say', 'Prefer to self-describe'}
[ ]
!pip install plotly
Collecting plotly
  Downloading plotly-5.4.0-py2.py3-none-any.whl (25.3 MB)
     |████████████████████████████████| 25.3 MB 16.9 MB/s 
Requirement already satisfied: six in /home/ivanyingx/miniconda3/lib/python3.9/site-packages (from plotly) (1.16.0)
Collecting tenacity>=6.2.0
  Downloading tenacity-8.0.1-py3-none-any.whl (24 kB)
Installing collected packages: tenacity, plotly
Successfully installed plotly-5.4.0 tenacity-8.0.1
[ ]
import plotly.express as px
px.histogram(df, "Q2", labels={"value": "Gender"}, title="Counts of Gender")

Cool! What about where the residencies of the individuals? We'll turn it up a notch and plot these on a world map, heating them by the number of respondants from that country. This is known as a choropleth map and will require us to change our country names into 3 letter ISO codes.

The first thing we need to do is look at the countries column itself (i.e. Q3). After doing so, it's worth updating values to something more conventional if they're not there yet.

We'll then load in a package to which we can pass a country and have it return the ISO code for us. Then we'll use the new column to plot our choropleth.

[ ]
  set(df["Q3"])
{'Algeria',
 'Argentina',
 'Australia',
 'Austria',
 'Bangladesh',
 'Belarus',
 'Belgium',
 'Brazil',
 'Canada',
 'Chile',
 'China',
 'Colombia',
 'Czech Republic',
 'Denmark',
 'Egypt',
 'France',
 'Germany',
 'Greece',
 'Hong Kong (S.A.R.)',
 'Hungary',
 'India',
 'Indonesia',
 'Iran, Islamic Republic of...',
 'Ireland',
 'Israel',
 'Italy',
 'Japan',
 'Kenya',
 'Malaysia',
 'Mexico',
 'Morocco',
 'Netherlands',
 'New Zealand',
 'Nigeria',
 'Norway',
 'Other',
 'Pakistan',
 'Peru',
 'Philippines',
 'Poland',
 'Portugal',
 'Republic of Korea',
 'Romania',
 'Russia',
 'Saudi Arabia',
 'Singapore',
 'South Africa',
 'South Korea',
 'Spain',
 'Sweden',
 'Switzerland',
 'Taiwan',
 'Thailand',
 'Tunisia',
 'Turkey',
 'Ukraine',
 'United Kingdom of Great Britain and Northern Ireland',
 'United States of America',
 'Viet Nam'}
So here are the values I think need updating:

Hong Kong (S.A.R.)
Iran, Islamic Republic of...
United Kingdom of Great Britain and Northern Ireland
Viet Nam
South Korea
Also notice there's an "Other"

[ ]
print("Percentage of 'Other':", df["Q3"].value_counts()["Other"]/len(df) * 100)

values_to_update = {"Q3": 
                    {"Hong Kong (S.A.R.)": "Hong Kong",
                     "Iran, Islamic Republic of...": "Iran",
                     "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                     "South Korea": "Republic of Korea",
                     "Viet Nam": "Vietnam"}}

## Using the replace method, update the values in the relevant column
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
df.replace(values_to_update, inplace=True)
set(df["Q3"])
Percentage of 'Other': 5.345640817568595
{'Algeria',
 'Argentina',
 'Australia',
 'Austria',
 'Bangladesh',
 'Belarus',
 'Belgium',
 'Brazil',
 'Canada',
 'Chile',
 'China',
 'Colombia',
 'Czech Republic',
 'Denmark',
 'Egypt',
 'France',
 'Germany',
 'Greece',
 'Hong Kong',
 'Hungary',
 'India',
 'Indonesia',
 'Iran',
 'Ireland',
 'Israel',
 'Italy',
 'Japan',
 'Kenya',
 'Malaysia',
 'Mexico',
 'Morocco',
 'Netherlands',
 'New Zealand',
 'Nigeria',
 'Norway',
 'Other',
 'Pakistan',
 'Peru',
 'Philippines',
 'Poland',
 'Portugal',
 'Republic of Korea',
 'Romania',
 'Russia',
 'Saudi Arabia',
 'Singapore',
 'South Africa',
 'Spain',
 'Sweden',
 'Switzerland',
 'Taiwan',
 'Thailand',
 'Tunisia',
 'Turkey',
 'Ukraine',
 'United Kingdom',
 'United States of America',
 'Vietnam'}
[ ]
!pip install pycountry
Collecting pycountry
  Downloading pycountry-20.7.3.tar.gz (10.1 MB)
     |████████████████████████████████| 10.1 MB 19.2 MB/s 
Building wheels for collected packages: pycountry
  Building wheel for pycountry (setup.py) ... done
  Created wheel for pycountry: filename=pycountry-20.7.3-py2.py3-none-any.whl size=10746863 sha256=274030825c7e6d6a70b51e7734fb13343e2e6210b7dbd229ca8fea06961a7c64
  Stored in directory: /home/ivanyingx/.cache/pip/wheels/de/cb/0e/b40fff1168704e2498630eee7ad70a78b458fe4a902179ae2c
Successfully built pycountry
Installing collected packages: pycountry
Successfully installed pycountry-20.7.3
[ ]
import pycountry

## Create a new dataframe which will hold only the unique countries, their country codes and the number of instances of this country - WITHOUT "Other"
countries = df["Q3"][df["Q3"]!= "Other"].unique()
countries_df = pd.DataFrame(countries, columns=["Country"])
countries_df["Count"] = countries_df["Country"].map(df["Q3"].value_counts())

## Create a new column in the dataframe which has the ISO country codes
country_codes = []
for country in countries_df["Country"]:
    country_code = pycountry.countries.search_fuzzy(country)[0] # Take the first element returned from the search
    country_codes.append(country_code.alpha_3)

countries_df["Country Code"] = country_codes
countries_df

[ ]
px.choropleth(countries_df, locations="Country Code", hover_name="Country", color="Count")

What about age by gender? We'll have to group variables together first.

[ ]
age_gender_df = df[["Q1", "Q2"]]
age_gender_groups = age_gender_df.groupby(["Q1", "Q2"]).size().unstack()
fig = px.bar(age_gender_groups, title="Count of Age per Gender", labels={"Q1": "Age", "value": "Count"})
fig.update_layout(legend_title_text='Gender')
# fig.update_layout(barmode="group")
fig.show()

So we see that the most frequent age of DSML employees are between 25-29. I see two reasons why these values are considerbly higher than the others:

Data Science and Machine Learning is a relatively new discipline, and now there exist direct education paths to these fields, which is more accessible to younger people
Think about where the data was collected from. Older people are perhaps less likely to use 'resource' sites like Kaggle becuase 1) They don't feel they need the learning experience and 2) Younger people are more common within social sites.
So far, we've just arbitrarily produced plots - perhaps a better plan is to perform a slightly more investigative analysis over the categories we outlined earlier. Let's do this with Education.

Education Analysis
Produce two plots:

The participants' formal education
The count of formal education per gender. Display this is a grouped bar chart
[ ]
fig = px.histogram(df, "Q4", height=800, title="Count of Education", labels={"value": "Education level"})
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()

[ ]
edu_gender_df = df[["Q2", "Q4"]]
edu_gender_groups = edu_gender_df.groupby(["Q4", "Q2"]).size().unstack()
fig = px.bar(edu_gender_groups, title="Education level count per Gender",
             labels={"Q2": "Education", "value": "Count"},
             height=800)
fig.update_layout(legend_title_text='Gender', xaxis={'categoryorder':'total descending'})
# fig.update_layout(barmode="group")
fig.show()

Let's create another diagram showing the same information but across 4 different plots

[ ]
fig = px.histogram(df, "Q4", 
                   facet_col="Q2", 
                   color="Q2",
                   title="Counts of Education level per Gender",
                   labels={"Q4": "Education Level"},
                   height=1000, 
                   facet_col_wrap=2, 
                   facet_col_spacing=0.1,
                   )
fig.update_layout(showlegend=False, xaxis={'categoryorder':'total descending'})
fig.update_yaxes(matches=None, showticklabels=True)
# fig.update_xaxes(showticklabels=True)
fig.show()

Some interesting things we can establish from this... That is:

Those who choose to self describe their gender are more likely have a doctorate than a bachelor's - vs every other category, which are more likely to have a bachelor's than a doctorate. Although if we note the counts, we can see that we're working with single digit figures - not something we can statistically extrapolate.
Those who preferred not to give their gender also preferred not to give their education level out either (relative to the other categories).
What I'm about to introduce next is probably one of my favourite kinds of plots. It's known as a Sankey Diagram.

The easiest way to start working with the Sankey diagram is to understand what we want as the terminating column. In this case, we'll use education level as the final column. We'll also need 'counts' - that is - how many total people are in each level of education. To save space on the diagram, we'll generalise some of the levels.

In this Sankey diagram, I want to visualise the path of gender, age, and country to education level.

[ ]
# We want five levels of education: Bachelor's, Master's, Doctoral, Professional, Other
## Create a new dataframe with just the education of surveyees where their education has been mapped to the above level
education_df = pd.DataFrame(df["Q4"])
education_df.rename(columns={"Q4": "Education Level"}, inplace=True)

values_to_update = {"Education Level": 
                    {"Some college/university study without earning a bachelor’s degree": "Other",
                     "No formal education past high school": "Other",
                     "I prefer not to answer": "Other"}}

education_df = education_df.replace(values_to_update)
set(education_df["Education Level"])
{'Bachelor’s degree',
 'Doctoral degree',
 'Master’s degree',
 'Other',
 'Professional degree',
 nan}
[ ]
# Let's drop na's from Education Level
education_df.isna().sum()
education_df = education_df.dropna(subset=["Education Level"])
education_df.isna().sum()
Education Level    0
dtype: int64
[ ]
## Add the gender, age and region columns to the new dataframe. Name the columns appropiately
cols_to_join = ["Q1", "Q2", "Q3"]
desired_col_names = ["Age", "Gender", "Region"]
for col, name in zip(cols_to_join, desired_col_names):
    education_df[name] = df[col]
    
education_df


[ ]
# For visualisation purposes let's create:
# 1. wider age bins as 18-29, 30-49, 50-69 and 70+
# 2. group genders as "Male", "Female", "Other"
# 3. Convert countries to continents - apart from "India", "United States of America" and "Other"

## Overwrite the age and gender columns so that ages are now: 18-29, 30-49, 50-69 and 70+ and genders are "Male", "Female" and "Other"
values_to_update = {
    "Age": {"18-21": "18-29", "22-24": "18-29", "25-29": "18-29",
            "30-34": "30-49", "35-39": "30-49", "40-44": "30-49", "45-49": "30-49",
            "50-54": "50-69", "55-59": "50-69", "60-69": "50-69"
           },
    "Gender": {"Prefer not to say": "Other", "Prefer to self-describe": "Other"}
}

education_df = education_df.replace(values_to_update)
education_df

[ ]
!pip install pycountry_convert
Collecting pycountry_convert
  Downloading pycountry_convert-0.7.2-py3-none-any.whl (13 kB)
Collecting repoze.lru>=0.7
  Downloading repoze.lru-0.7-py3-none-any.whl (10 kB)
Collecting pytest-mock>=1.6.3
  Downloading pytest_mock-3.6.1-py3-none-any.whl (12 kB)
Collecting pytest>=3.4.0
  Downloading pytest-6.2.5-py3-none-any.whl (280 kB)
     |████████████████████████████████| 280 kB 21.1 MB/s 
Requirement already satisfied: wheel>=0.30.0 in /home/ivanyingx/miniconda3/lib/python3.9/site-packages (from pycountry_convert) (0.36.2)
Requirement already satisfied: pycountry>=16.11.27.1 in /home/ivanyingx/miniconda3/lib/python3.9/site-packages (from pycountry_convert) (20.7.3)
Collecting pytest-cov>=2.5.1
  Downloading pytest_cov-3.0.0-py3-none-any.whl (20 kB)
Collecting pprintpp>=0.3.0
  Downloading pprintpp-0.4.0-py2.py3-none-any.whl (16 kB)
Requirement already satisfied: attrs>=19.2.0 in /home/ivanyingx/miniconda3/lib/python3.9/site-packages (from pytest>=3.4.0->pycountry_convert) (21.2.0)
Collecting pluggy<2.0,>=0.12
  Downloading pluggy-1.0.0-py2.py3-none-any.whl (13 kB)
Requirement already satisfied: packaging in /home/ivanyingx/miniconda3/lib/python3.9/site-packages (from pytest>=3.4.0->pycountry_convert) (21.3)
Collecting toml
  Downloading toml-0.10.2-py2.py3-none-any.whl (16 kB)
Collecting py>=1.8.2
  Downloading py-1.11.0-py2.py3-none-any.whl (98 kB)
     |████████████████████████████████| 98 kB 1.4 MB/s 
Collecting iniconfig
  Downloading iniconfig-1.1.1-py2.py3-none-any.whl (5.0 kB)
Collecting coverage[toml]>=5.2.1
  Downloading coverage-6.2-cp39-cp39-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (215 kB)
     |████████████████████████████████| 215 kB 151.1 MB/s 
Collecting tomli
  Downloading tomli-1.2.2-py3-none-any.whl (12 kB)
Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /home/ivanyingx/miniconda3/lib/python3.9/site-packages (from packaging->pytest>=3.4.0->pycountry_convert) (3.0.6)
Installing collected packages: tomli, toml, py, pluggy, iniconfig, coverage, pytest, repoze.lru, pytest-mock, pytest-cov, pprintpp, pycountry-convert
Successfully installed coverage-6.2 iniconfig-1.1.1 pluggy-1.0.0 pprintpp-0.4.0 py-1.11.0 pycountry-convert-0.7.2 pytest-6.2.5 pytest-cov-3.0.0 pytest-mock-3.6.1 repoze.lru-0.7 toml-0.10.2 tomli-1.2.2
[ ]
import pycountry_convert as pc
## Map countries to their relevant continents, unless the country is India, United States of America, or Other
countries_to_not_map = ["India", "United States of America", "Other"]
countries_to_map_to_continents = set(education_df["Region"])
for country in countries_to_not_map:
    countries_to_map_to_continents.discard(country)

countries_continent_dict = dict()
for country in countries_to_map_to_continents:
    country_alpha2 = pycountry.countries.search_fuzzy(country)[0].alpha_2
    continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    continent_name = pc.convert_continent_code_to_continent_name(continent_code)
    countries_continent_dict[country] = continent_name

to_update = {"Region": countries_continent_dict}
education_df = education_df.replace(to_update)
education_df

[ ]
# Re-indexing the columns in the order we want them for the diagram because it'll be easier to work with
education_df = education_df.reindex(["Gender", "Age", "Region", "Education Level"], axis=1)

col_names = education_df.columns.tolist()
node_labels = []
num_categorical_vals_per_col = []
for col in col_names:
    uniques = education_df[col].unique().tolist()
    node_labels.extend(uniques)
    num_categorical_vals_per_col.append(len(uniques))
    
node_labels, num_categorical_vals_per_col
(['Male',
  'Female',
  'Other',
  '18-29',
  '30-49',
  '50-69',
  '70+',
  'Europe',
  'India',
  'Oceania',
  'United States of America',
  'Asia',
  'Other',
  'South America',
  'Africa',
  'North America',
  'Master’s degree',
  'Professional degree',
  'Bachelor’s degree',
  'Other',
  'Doctoral degree'],
 [3, 4, 9, 5])
The num_categorical_vals_per_col is going to allow us to know which values from the previous we need to map to the next.

Now we need to construct our link dictionary. This is a bit less straightforward than the above. Our link dictionary will contain 3 lists: source, target and value. source and target indicate which nodes we want to connect to each other, and value indicates the quantity we want to 'fill' that connection with. source and target are numerical indexes of the node_labels list we created above.

For each category per column (source category), we're going to link that category to all the other categories of the next column (target category), with the size of how many of the source categories map to the target categories.

[ ]
education_df.groupby(["Gender", "Age"]).size()["Female"]["18-29"]
1857
[ ]
import numpy as np
import random

source = []
target = []
value = []
colors = []
for i, num_categories in enumerate(num_categorical_vals_per_col):
    
    if i == len(num_categorical_vals_per_col)-1:
        break
    
    # index allows us to refer to the categories by index from the `node_labels` list
    start_index = sum(num_categorical_vals_per_col[:i])
    start_index_next = sum(num_categorical_vals_per_col[:i+1])
    end_index_next = sum(num_categorical_vals_per_col[:i+2])
#     print(start_index, start_index_next, end_index_next)
    
    # i can also give us the category column to refer to
    col_name = col_names[i]
    next_col_name = col_names[i+1]
    
    grouped_df = education_df.groupby([col_name, next_col_name]).size()
#     print(grouped_df)
    
    for source_i in range(start_index, start_index_next):
        for target_i in range(start_index_next, end_index_next):
            source.append(source_i)
            target.append(target_i)
            source_label = node_labels[source_i]
            target_label = node_labels[target_i]
            # if the index doesn't exist in the grouped_df, then the value is 0
            try:
                value.append(grouped_df[source_label][target_label])
            except:
                value.append(0)
            
            random_color = list(np.random.randint(256, size=3)) + [random.random()]
            random_color_string = ','.join(map(str, random_color))
            colors.append('rgba({})'.format(random_color_string))

print(source)
print(target)
print(value)

link = dict(source=source, target=target, value=value, color=colors)
[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15]
[3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 7, 8, 9, 10, 11, 12, 13, 14, 15, 7, 8, 9, 10, 11, 12, 13, 14, 15, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20]
[8296, 6204, 1245, 79, 1857, 1132, 155, 4, 164, 138, 33, 16, 1760, 3596, 88, 1091, 1881, 500, 468, 673, 260, 1968, 1047, 175, 1456, 1227, 450, 585, 288, 278, 409, 63, 52, 453, 176, 64, 103, 27, 86, 16, 3, 4, 40, 5, 15, 4, 1, 11, 2199, 119, 650, 297, 888, 1683, 204, 2321, 250, 251, 151, 6, 84, 24, 54, 1520, 31, 702, 156, 631, 1452, 74, 1023, 312, 428, 448, 28, 275, 127, 151, 482, 91, 344, 102, 141, 364, 38, 407, 74, 106, 250, 20, 187, 61, 117]
[ ]
import plotly.graph_objects as go

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = node_labels,
      color = "blue"
    ),
    link = link)])

fig.update_layout(title_text="Sankey Diagram (Gender, Age, Region, Education)", font_size=10)
fig.show()

Age
We took a very quick look at age earlier on, but I'm interested in finding out some more things about it. Using the orignal df, produce the following plots:

A facet plot of the count of education levels per age
A facet plot of the different roles by age
We will then cover the following:

A plot of the count of the different languages
Subplots/facet plot of the count of different languages per age
[ ]
# We'll sort the df by Age so our plot displays in the age order
df = df.sort_values(by=["Q1"])

fig = px.histogram(df, "Q4", facet_col="Q1",
             color="Q1",
             title="Counts of Education level per Age",
             labels={"Q1": "Age", "Q4": "Education Level"},
             height=1000, 
             facet_col_wrap=4, 
             facet_col_spacing=0.1)

fig.update_layout(showlegend=False, xaxis={'categoryorder':'total descending'})
fig.update_yaxes(matches=None, showticklabels=True)

Interesting! The younger results are perhaps what we'd expect - 18-21 year olds typically aren't old enough to do masters degrees hence the number of bachelor's is higher for them. However, for almost every other age group, Master's degrees are prominent. Curiously, those over 70 are more likely to have a doctorate.

[ ]
set(df["Q5"])
{'Business Analyst',
 'DBA/Database Engineer',
 'Data Analyst',
 'Data Engineer',
 'Data Scientist',
 'Not employed',
 'Other',
 'Product/Project Manager',
 'Research Scientist',
 'Software Engineer',
 'Statistician',
 'Student',
 nan}
[ ]
fig = px.histogram(df, "Q5", facet_col="Q1",
             color="Q1",
             title="Counts of Education level per Age",
             labels={"Q1": "Age", "Q5": "Job Role"},
             height=2000, 
             facet_col_wrap=2, 
             facet_col_spacing=0.1)

fig.update_layout(showlegend=False)
fig.update_xaxes(showticklabels=True, tickangle=45)
fig.update_yaxes(matches=None, showticklabels=True)

[ ]
# Our data is in columns 18_p1, 18_p12
# Our first step will be to create a new column called "Known programming languages", and per row, create a comma separated list which contain the programming languages they know (obviously excluding NaNs)
programming_cols = ["Q18_Part_{}".format(str(i)) for i in range(1, 13)]
programming_df = df[programming_cols]
programming_df

[ ]
programming_col = []
for row in programming_df.itertuples(index=False):
    languages_known = [language for language in row if isinstance(language, str)]
    programming_col.append(",".join(languages_known))
    
programming_df["languages_known"] = programming_col
programming_df

[ ]
# Let's trim the new df so it only has our new col
programming_df.drop(labels=programming_cols, axis=1, inplace=True)
programming_df

[ ]
# Assume blanks mean they don't know a language and replace both the blanks and "None" with "None/NA"
values_to_update = {"languages_known": {"": "None/NA", "None": "None/NA"}}
programming_df = programming_df.replace(values_to_update)
programming_df

[ ]
# Let's use the get_dummies method to create new columns 
language_dummies = programming_df['languages_known'].str.get_dummies(sep=',')
language_dummies

[ ]
fig = px.bar(language_dummies.sum(), labels={"index": "Programming Language"}, title="Count of Programming Languages")
fig.update_layout(showlegend=False, xaxis={'categoryorder':'total descending'})
fig.show()

[ ]
# For 4. let's get the ages from the original dataframe and join them to this new dataframe
ages = df["Q1"]
language_dummies_with_age = language_dummies.join(ages).rename(columns={"Q1": "Age"})
language_dummies_with_age

[ ]
programming_languages_by_age = language_dummies_with_age.groupby(["Age"]).sum()
px.bar(programming_languages_by_age)
# px.bar(programming_languages_by_age.T)
programming_languages_by_age = programming_languages_by_age.reindex(
    programming_languages_by_age.mean().sort_values().index, axis=1)
programming_languages_by_age = programming_languages_by_age.iloc[:, ::-1]
programming_languages_by_age

[ ]
programming_languages_by_age_row_norm = programming_languages_by_age.div(programming_languages_by_age.sum(axis=1), axis=0)
[ ]
from plotly.subplots import make_subplots

# programming_languages_by_age.index
programming_languages = programming_languages_by_age_row_norm.columns.tolist()
fig = make_subplots(4, 3, subplot_titles=programming_languages_by_age_row_norm.index)
for i, age_range in enumerate(programming_languages_by_age_row_norm.index):
    row = (i // 3) + 1
    col = (i % 3) + 1
    fig.add_trace(
        go.Bar(x=programming_languages, y=programming_languages_by_age_row_norm.iloc[i]),
        row=row, col=col
    )
fig.update_layout(showlegend=False, height=1000, title="Percent of Known Programming Languages by Age")
fig.update_yaxes(tickformat="%")
fig.show()

Looks like everyone likes Python! Younger people (who are most likely to be doing a bachelor's) have a relatively higher percentage of lower level languages like C, C++ and Java. This could be because they have to study these languages at university. As the data scientists specialise more in their career they seem to move away from these languages into more typical DSML related languages. The 60-69 group has a high level of R users relative to the other age groups, while it seems like most people over 70 don't know any programming languages.

Perhaps more useful to us is what programming languages are popular against what job roles (Q5). Create a plot which demonstrates this

[ ]
## Join the Job Titles question with the language_dummies dataframe
language_dummies_with_job = language_dummies.join(df["Q5"]).rename(columns={"Q5": "Job Title"})
language_dummies_with_job

[ ]
## Group by Job Title, and aggregate the number, normalize each row, and sort the dataframe on the mean of the columns
languages_job_title_grouped = language_dummies_with_job.groupby(["Job Title"]).sum()
languages_job_title_grouped = languages_job_title_grouped.div(languages_job_title_grouped.sum(axis=1), axis=0)
languages_job_title_grouped = languages_job_title_grouped.reindex(
    languages_job_title_grouped.mean().sort_values().index, axis=1)
languages_job_title_grouped = languages_job_title_grouped.iloc[:, ::-1]
languages_job_title_grouped

[ ]
px.imshow(languages_job_title_grouped, title="Heatmap of Programming Languages and Job Title")

[ ]
## Plot the data!
programming_languages = languages_job_title_grouped.columns.tolist()
fig = make_subplots(4, 3, subplot_titles=languages_job_title_grouped.index)
for i, role in enumerate(languages_job_title_grouped.index):
    row = (i // 3) + 1
    col = (i % 3) + 1
#     numbers = langugages_job_title_grouped.iloc[i]
#     as_percent = [number / sum(numbers) for number in numbers]
    fig.add_trace(
        go.Bar(x=programming_languages, y=languages_job_title_grouped.iloc[i]),
        row=row, col=col
    )
fig.update_layout(showlegend=False, height=1000, title="Percent of Known Programming Languages Usage per Job Role")
fig.update_yaxes(tickformat="%")
fig.show()

Findings:

Python is averagly the most popular language amongst all job roles
SQL is the most popular language amongst database engineers
MATLAB is relatively more popular amongst research scientists than other job roles
Statistician's prefer R over Python
Student's and oftware engineers prefer C++ over R
Let's do one more plot - a heatmap of the framework that each job role likes to use

[ ]
## Plot the top 3 frameworks that each job role likes to use
# 28p1 - 28p12
## Create a dateframe which contains just the framework columns
framework_cols = ["Q28_Part_{}".format(str(i)) for i in range(1, 13)]
framework_df = df[framework_cols]
framework_df
#%%
#Create a general function, get_df_for_dummies(), which takes in a dataframe, a column name prefix string and an upper range, and returns a dataframe populated with the range of columns based on the prefix string
def get_df_for_dummies(df, prefix_string, end_range, start_range=1):
    ## HINT: use a range function and iteration to genereate the column names
    dummies_cols = [prefix_string + str(i) for i in range(start_range, end_range)]
    ## HINT: extract and return the column names from the dataframe
    return df[dummies_cols]
    
## Create a column in this dataframe called 'frameworks used', and populate that column by comma separated frameworks
framework_col = []
for row in framework_df.itertuples(index=False):
    frameworks_used = [framework for framework in row if isinstance(framework, str)]
    framework_col.append(",".join(frameworks_used))
    
framework_df["frameworks_used"] = framework_col
framework_df

## Replace blank and None columns with "None/NA"
values_to_update = {"frameworks_used": {"": "None/NA", "None": "None/NA"}}
framework_df = framework_df.replace(values_to_update)
framework_df

# %%
# Create a general function, get_dummies(), which takes in a dataframe and returns a column populated with comma separated values of each of the individual values over the dataframe.

def get_dummies_col(df, sep=","):

    ## initialise an empty list to hold the column of strings which will be used to create a dummies dataframe
    dummies_col = []
    
    ## iterate over each of the rows in the dataframe, in a manner where we can access the individual elements of the cells
    ## get a list of the values of the cells over the row (e.g. a list of programming languages). Make sure you don't add nan's to your 
    ## join these as a comma separated string, and append it to the empty list for the column
    for row in df.itertuples(index=False):
        values = [item for item in row if isinstance(item, str)]
        dummies_col.append(sep.join(values))
        
    ## create a new column in the dataframe called "dummies", which takes on the contents of dummies column
    df["dummies"] = dummies_col
    
    ## replace all the "" and "None"s from from the dataframe with "None/NA"
    values_to_update = {"dummies": {"": "None/NA", "None": "None/NA"}}
    df = df.replace(values_to_update)
    
    ## return the new dataframe
    return df
# %%
## Create a dummies dataframe for the frameworks
framework_dummies = framework_df['frameworks_used'].str.get_dummies(sep=',')
framework_dummies
# %%
def dummies_from_series(series, sep=","):
    ## return a dummies dataframe from the dataframe. 
    # Remember which column was used to assign the strings we want to create dummies over 
    return series["dummies"].str.get_dummies(sep=sep)
    
## Create a new dataframe which join this dataframe with job roles
frameworks_for_job_role = framework_dummies.join(df["Q5"]).rename(columns={"Q5": "Job Title"})
frameworks_for_job_role

## Group the dataframe by the job title, and aggregate over the programming languages
frameworks_for_job_role_grouped = frameworks_for_job_role.groupby(["Job Title"]).sum()
frameworks_for_job_role_grouped

# Create a general function, group_dummies_by() which takes in a dummies dataframe, and a Series, and returns the dummies grouped by and aggregated by the Series.

def group_dummies_by(dummies_df, series):
    series_name = series.name
    to_group = dummies_df.join(series)
    grouped = to_group.groupby([series_name]).sum()
    
    return grouped
    
framework_df = get_df_for_dummies(df, "Q28_Part_", 13)
framework_df = get_dummies_col(framework_df)
framework_dummies = dummies_from_series(framework_df)
frameworks_for_job_role_grouped = group_dummies_by(framework_dummies, df["Q5"])
frameworks_for_job_role_grouped = frameworks_for_job_role_grouped.div(
    frameworks_for_job_role_grouped.sum(axis=1), axis=0)
frameworks_for_job_role_grouped.index.rename("Job Role", inplace=True)
frameworks_for_job_role_grouped

## Produce a heatmap of the above dataframe!
px.imshow(frameworks_for_job_role_grouped, title="Heatmap of preferred Frameworks per Job Role")
# # %%
# Findings:

# None/NA may be slightly misleading. It doesn't always imply that their role has no need for frameworks because it also incorporates people who may not have answered the question because their tool of use wasn't provided as an option. For example, a statistician may beusing an R framework who's option isn't provided here.
# Scikit-learn is the most popular listed tool
# Statistician's seem to use random forest a lot
#%%
# Yearly Compensation
# Let's start considering our target variable now! We'll start basic and work our way up. Let's plot a histogram of salary earnings, and sort have this plot in order of salary ranges

px.histogram(df, "Q10", labels={"Q10": "Salary"}, title="Count of salary ranges (Unsorted)")

# I can already see some interesting findings from this, but before I mention anything I want us to sort the x axis in order of numerical value. That is, the leftmost column is $0-999 and the rightmost is > $500,000.
# Spend some time thinking and implementing how you would solve this problem. If after a few minutes you aren't able to think/come up with a solution, read and implement the spoiler below
# Create a new dataframe with just the salarys
# Get the set of salaries
# Create a mapping of the salary categories to an int, where the int is the first numerical part of the string (e.g. {"$0-999": 0, "100,000-124,999": 100000}). This part will require you to use the `.replace()` and `.split()` methods native to Python
# Replace the salaries in the dataframe with the integer value
# Sort the dataframe in ascending numerical order
# Reverse the maping and replace the ints with their string variant
# Plot the dataframe, and replace the x labels with the salary strings

salary_df = pd.DataFrame(df["Q10"])
salary_df.rename(columns={"Q10": "Salary"}, inplace=True)
salary_set = set(salary_df["Salary"])
salary_string_int_dict = dict()

for string_salary in salary_set:
    
    if isinstance(string_salary, float): continue
        
    salary = string_salary.replace("$", "").replace("> ", "").replace(",", "")
    salary = salary.split("-")[0]
    salary_string_int_dict[string_salary] = int(salary)

values_to_update = {"Salary": salary_string_int_dict}
salary_df = salary_df.replace(values_to_update)
salary_df = salary_df.sort_values("Salary")

salary_int_string_dict = {v:k for k,v in salary_string_int_dict.items()}
values_to_update = {"Salary": salary_int_string_dict}
salary_df = salary_df.replace(values_to_update)

percent_na = np.round(100 * salary_df["Salary"].isna().sum()/len(salary_df), 2)
print("Percent of users who didn't answer the salary question:", percent_na)
px.histogram(salary_df, "Salary", title="Count of Salary ranges")
#%%
# Almost 37% of the survey participants explicitly didn't answer this question. There seems to be an oddly large amount of people who are earning betweeen \$0 and \$999 a year. I suspect that this is so high because a lot of people who didn't want to answer the question (and didn't realise it was optional) ticked this box. We can also spot some other interesting facts - that is - we seem to have "two" peaks at vastly different salaries - one at 10,000 - 14,999 and the other at 100,000 - 124,999. Can you think why this could be? Also it looks like there are some very rich kagglers, with 83 of them earning over \$500k a year.
# The top 3 most popular wages (par 0 - 999) seem to be 10,000 - 14,999, 100,000 - 124,999 and 30,000 - 39,999. Produce a choropleth plot of the median salary of the countries so we can better discern in what regions we can expect to earn what
## Produce a choropleth plot of the median salary
median_salaries_df = df[["Q3", "Q10"]]
median_salaries_df.rename(columns={"Q3": "Country", "Q10": "Salary"}, inplace=True)
values_to_update = {"Salary": salary_string_int_dict}
median_salaries_df = median_salaries_df.replace(values_to_update)
median_salaries_df = median_salaries_df.groupby(["Country"]).median()
median_salaries_df
#%%
country_codes = []
for country in median_salaries_df.index:
    country_code = pycountry.countries.search_fuzzy(country)[0] # Take the first element returned from the search
    country_codes.append(country_code.alpha_3)

median_salaries_df["Country Code"] = country_codes
median_salaries_df
# %%
salaries_series = median_salaries_df["Salary"]
values_to_update = {"Salary": salary_int_string_dict}
median_salaries_df = median_salaries_df.replace(values_to_update)
median_salaries_df["Salary Values"] = salaries_series
#%%
px.choropleth(median_salaries_df, locations="Country Code", hover_name=median_salaries_df.index, color="Salary Values", hover_data=["Salary"], title="Median Salaries by Country")

#%%
# Considering the amount of participants from USA and India that we discerned earlier on, we can take these values to be more correct to the underlying data generation distribution than most other countries. Working under this assumption, it looks like in the US and Switzerland, it would be reasonable to be on 100,000+, whereas those in India would most likely be earning something in the 7,500 - 9,999 range. It seems Australia has some high paying jobs too. We'd expect the average salary in the UK to be higher than the 10,000 - 14,999 mark. What are some things we could investigate to identify why this salary value is lower than we expect it to be?
# What about salaries by gender?

salaries_by_gender_df = df[["Q2", "Q10"]]
salaries_by_gender_df.rename(columns={"Q2": "Gender", "Q10": "Salary"}, inplace=True)
values_to_update = {"Salary": salary_string_int_dict}
salaries_by_gender_df = salaries_by_gender_df.replace(values_to_update)
salaries_by_gender_df

px.box(salaries_by_gender_df, "Gender", "Salary", labels={"Salary": "Salary (Lower Bound)"}, title="Boxplot of Salary per Gender")

# Even though it looks like any of the categories are able to reach the highest salary bracket, it seems that females have the lowest median salary. Those who prefer to self describe seem to be more likely to have higher average salaries too.

# What are the best paying jobs? Let's plot the mean salary of each role
#%%
salaries_job_df = df[["Q5", "Q10"]]
salaries_job_df.rename(columns={"Q5": "Job Title", "Q10": "Salary"}, inplace=True)

values_to_update = {"Salary": salary_string_int_dict}
salaries_job_df = salaries_job_df.replace(values_to_update)
salaries_job_df

# salary_series = salaries_job_df["Salary"]
# salaries_job_df["Salary Bracket"] = salary_series
# salaries_job_df

grouped_mean_salaries = salaries_job_df.groupby(["Job Title"]).mean().reset_index().sort_values(by="Salary", ascending=False)
grouped_mean_salaries.dropna(inplace=True)
px.bar(grouped_mean_salaries, "Job Title", "Salary", labels={"Salary": "Mean Salary"}, title="Mean Salary per Job Role")

# What could perhaps be a more informative way to represent these salaries? 
# I'm thinking boxplot. Let's drop the NAs and "0" salaries and visualise this

salaries_job_df.dropna(inplace=True)
salaries_job_df = salaries_job_df[(salaries_job_df["Salary"] != 0)]
fig = px.box(salaries_job_df, "Job Title", "Salary", labels={"Salary": "Salary (Lower Bound of Bracket)"})
fig.show()

# Findings:

# The survey seems to indicate that all job roles have the potential to gain a job at \$500k+ apart from that of a Database Engineer
# Globally, Software Engineers and Data Analysts have the two lowest paying average salaries - with a software engineer having a lower median salary than a data analyst, but a higher mean (as seen from the first chart)
# Product/Project Management and Data Science seem to be the most lucrative job roles to be in - with the former slightly taking the lead
# I wonder the percent of applicable Data Scientists who earn above \$500k vs Project Managers:

#%%
num_data_scientists_above_500 = len(salaries_job_df[(salaries_job_df["Job Title"] == "Data Scientist") & (salaries_job_df["Salary"] == 500000)])
num_project_managers_above_500 = len(salaries_job_df[(salaries_job_df["Job Title"] == "Product/Project Manager") & (salaries_job_df["Salary"] == 500000)])

percent_ds_above_500 = 100 * num_data_scientists_above_500/len(salaries_job_df)
percent_pm_above_500 = 100 * num_project_managers_above_500/len(salaries_job_df)

print("The percent of Data Scientists who earn above $500,000: {}%".format(np.round(percent_ds_above_500, 2)))
print("The percent of Project Managers who earn above $500,000: {}%".format(np.round(percent_pm_above_500, 2))) 

# Let's see how much effect years of programming (Q11) have against salary
programming_experience_salary_df = df[["Q10", "Q15"]]
programming_experience_salary_df.rename(columns={"Q10": "Salary", "Q15": "Programming Experience"}, inplace=True)
values_to_update = {"Salary": salary_string_int_dict}
programming_experience_salary_df = programming_experience_salary_df.replace(values_to_update)
programming_experience_salary_df

category_array = ["I have never written code", "< 1 years", "1-2 years", "3-5 years", "5-10 years", "10-20 years", "20+ years"]
# fig = px.scatter(programming_experience_salary_df, "Programming Experience", "Salary", title="Density of Programming Experience vs Salary")
fig = px.scatter(programming_experience_salary_df, "Programming Experience", "Salary", facet_col=df["Q2"],title="Density of Programming Experience vs Salary")
fig.update_traces(marker=dict(
            opacity=0.05,
            size=20,
            line=dict(
                color='MediumPurple',
                width=0.5
            )))
fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray':category_array})
fig.show()

# Findings:

# There are people earning \$500k+ who haven't ever written code. Same with people who have under a years worth of programming experience.
# There is a trend however with more experienced programmers earning a higher salary across all levels of salary brackets
# There aren't that many experienced females (i.e. more than 10-20 years experience)
# Now we're going to plot another Sankey Diagram - this time tracking how Gender, Age, Degree, Role and Country all have an affect on your Salary. To achieve this, generalise the code I wrote earlier on for the Sankey diagram into a function, get_sankey_data, which takes a reindexed dataframe as argument and returns the node_labels and link dictionary. Subsequently, use these items to plot a Sankey diagram
#%%
## Function-ize the sankey diagram code I wrote earlier
def get_sankey_data(reindexed_df):
    col_names = reindexed_df.columns.tolist()
    node_labels = []
    num_categorical_vals_per_col = []
    
    for col in col_names:
        uniques = reindexed_df[col].unique().tolist()
        node_labels.extend(uniques)
        num_categorical_vals_per_col.append(len(uniques))


    source = []
    target = []
    value = []
    for i, num_categories in enumerate(num_categorical_vals_per_col):

        if i == len(num_categorical_vals_per_col)-1:
            break

        # index allows us to refer to the categories by index from the `node_labels` list
        start_index = sum(num_categorical_vals_per_col[:i])
        start_index_next = sum(num_categorical_vals_per_col[:i+1])
        end_index_next = sum(num_categorical_vals_per_col[:i+2])


        # i can also give us the category column to refer to
        col_name = col_names[i]
        next_col_name = col_names[i+1]

        grouped_df = reindexed_df.groupby([col_name, next_col_name]).size()

        for source_i in range(start_index, start_index_next):
            for target_i in range(start_index_next, end_index_next):
                source.append(source_i)
                target.append(target_i)
                source_label = node_labels[source_i]
                target_label = node_labels[target_i]
                # if the index doesn't exist in the grouped_df, then the value is 0
                try:
                    value.append(grouped_df[source_label][target_label])
                except:
                    value.append(0)

#                 random_color = list(np.random.randint(256, size=3)) + [random.random()]
#                 random_color_string = ','.join(map(str, random_color))
#                 colors.append('rgba({})'.format(random_color_string))

    link = dict(source=source, target=target, value=value)
    return node_labels, link
    
#%%
## Create a new dataframe with the relevant variables we want to plot our Sankey with, re-index it, and pass it to the get_sankey_data function
salaries_sankey_df = df[["Q1", "Q2", "Q3", "Q4", "Q5", "Q10"]]
salaries_sankey_df.rename(columns={"Q1": "Age", "Q2": "Gender", "Q3": "Country", "Q4": "Education", "Q5": "Role", "Q10": "Salary"}, inplace=True)
salaries_sankey_df = salaries_sankey_df.reindex(["Gender", "Age", "Education", "Role", "Country", "Salary"], axis=1)
node_labels, link = get_sankey_data(salaries_sankey_df)
node_labels

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = node_labels,
      color = "blue"
    ),
    link = link)])

fig.update_layout(title_text="Sankey Diagram (Gender, Age, Education, Role, Country, Salary)", font_size=10, height=1000)
fig.show()

# The above diagram is really messy... Let's clean it up as follows:

# Group countries by continent (again we'll leave India and USA as is)
# Create wider bins for salary (maybe 5 bins?), dropping \$0-999
# Add in coding experience as a column in the above plot
#%%
## Replace countries with continents
values_to_update = {"Country": countries_continent_dict}
salaries_sankey_df = salaries_sankey_df.replace(values_to_update)
salaries_sankey_df

## Drop rows with $0-999 and create wider bins for salary
set(salaries_sankey_df["Salary"])
salaries_sankey_df = salaries_sankey_df[salaries_sankey_df["Salary"] != "$0-999"]
under_30000, under_80000, under_150000, under_300000, under_500000 = "0 - 29,999", "30,000 - 79,999", "80,000 - 149,999", "150,000 - 299,999", "300,000 - 500,000" 
salary_wider_bins_dict = dict()
for salary_string in set(salaries_sankey_df["Salary"]):
    
    if salary_string == "> $500,000" or isinstance(salary_string, float):
        continue
    
    salary_upper_bound = salary_string.split("-")[-1]
    salary_upper_bound = salary_upper_bound.replace(",", "")
    salary_upper_bound = int(salary_upper_bound)
    
    if salary_upper_bound < 30000:
        salary_wider_bins_dict[salary_string] = under_30000
    elif salary_upper_bound < 80000:
        salary_wider_bins_dict[salary_string] = under_80000
    elif salary_upper_bound < 150000:
        salary_wider_bins_dict[salary_string] = under_150000
    elif salary_upper_bound < 300000:
        salary_wider_bins_dict[salary_string] = under_300000
    elif salary_upper_bound < 500000:
        salary_wider_bins_dict[salary_string] = under_500000

values_to_update = {"Salary": salary_wider_bins_dict}
salaries_sankey_df = salaries_sankey_df.replace(values_to_update)
salaries_sankey_df

## Create a new column in salaries_sankey_df which is the programming experience length
salaries_sankey_df["Programming Experience"] = df["Q15"]
salaries_sankey_df

## Reindex and plot the Sankey diagram!
# Drop the Age and Gender columns and reindex the dataframe as:
# Role, Programming Experience, Region, Education, Salary
salaries_sankey_df = salaries_sankey_df.rename(columns={"Country": "Region"})
salaries_sankey_df = salaries_sankey_df.reindex(["Role", "Programming Experience", "Region", "Education", "Salary"], axis=1)
node_labels, link = get_sankey_data(salaries_sankey_df)
fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = node_labels,
      color = "blue"
    ),
    link = link)])

fig.update_layout(title_text="Sankey Diagram (Role, Programming Experience, Region, Education, Salary)", font_size=10, height=800)
fig.show()

# The information and the paths to the different salary brackets becomes a lot clearer to interpret now. Some interesting findings that I've found from this graph is that there are some software engineers and data scientists who have never written code before - I'm unsure how this could be the case but they are small in number so we could chalk it up to anomalies. The majority of people earning over 150,000 seem to hold either master's or doctoral degrees. Playing around with the order of indexing can quickly allow you to draw insights from different combinations of columns (e.g. by putting Education before Job Role, we can see what roles people are likely to go into based on their education).

# One last plot! Question 9 asks about the skills and responsibilities that their current job entails. Produce subplots of these skills, aggregated, for each of the salary brackets. Use the 6 salary brackets we defined previously. Ensure that the columns of the skills dataframe you create are intact, and demonstrate your plot/and any other python output in an interpretable manner - whether this be the strings on the axis or the ordering of the facets.

skills_df = get_df_for_dummies(df, "Q9_Part_", 9)
skills_df = get_dummies_col(skills_df, sep="::")
skills_dummies = dummies_from_series(skills_df, sep="::")
skills_grouped = group_dummies_by(skills_dummies, salaries_sankey_df["Salary"])
skills_grouped = skills_grouped.reindex(["0 - 29,999", "30,000 - 79,999", "80,000 - 149,999", "150,000 - 299,999", "300,000-500,000", "> $500,000"])
skills_grouped

skills_df_columns = skills_grouped.columns
skills_df_columns_mapping = {col: i for i, col in enumerate(skills_df_columns)}
skills_grouped = skills_grouped.rename(columns=skills_df_columns_mapping)
[print(i, "\t", col) for col, i in skills_df_columns_mapping.items()]
print()

## Produce plots! The x axis should be top 4 programming languages, and the y axis the count of them.
fig = make_subplots(2, 3, subplot_titles=skills_grouped.index)
for i, role in enumerate(skills_grouped.index):
    row = (i // 3) + 1
    col = (i % 3) + 1
    skills_values = skills_grouped.iloc[i]
    fig.add_trace(
        go.Bar(x=skills_values.index, y=skills_values),
        row=row, col=col
    )
    fig.update_xaxes(type="category")
fig.update_layout(showlegend=False, height=800, title="Skills/Responsibilities per Salary Bracket")
fig.show()


# Findings:

# Across all salary brackets, 0, "Analyze and understand data to influence product or business decisions" is present.
# Apart from the lowest and highest bracket, it looks like a lot of jobs involve 3, builing prototypes which explores applying machine learning to new areas.
# The top two salary brackets have a higher relative frequency of answer 4 - Do research that advances the state of the art of machine learning.
# The last three brackets seem to have a higher proportion of employess who have option 5 (Experimentation and iteration to improve existing ML models) as a responsibility.
# Option 7, None/NA, seems to be more prevalent in the bottom two brackets and the uppermost bracket. I suspect the reason for this is that none of the job descriptions apply to these individuals, whereas the other three brackets have 
# jobs responsibilities focused more on data science and machine learning.