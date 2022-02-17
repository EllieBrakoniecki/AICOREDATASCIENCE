#%%
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as ss
import matplotlib.pylab as plt
data= pd.read_csv("https://aicore-files.s3.amazonaws.com/Data-Science/ab_data.csv")
# %%
data

# %%
plt.figure("Test Plots")
sns.distplot(data.Conversion_A, label='A')
sns.distplot(data.Conversion_B, label='B')
plt.legend()
plt.show()
# %%
t_stat, p_val= ss.ttest_ind(data.Conversion_B, data.Conversion_A)
print(f'The t-test value is {t_stat}, and the p-value is {p_val}')
# Here, our p-value is less than the significance level i.e 0.05. 
# Hence, we can reject the null hypothesis. This means that in our A/B testing, option B is performing better than option A. So our recommendation would be to replace our current option with B to bring more traffic on our website.


# %%
# Practical - Non-parametric Tests 
# Download the data in the following link: https://aicoreassessments.s3.amazonaws.com/amazon_data.csv 
# The data contains the offset delivery time of some Amazon deliveries.
# The Null Hypothesis is that the median offset time is 30 mins, and the Alternative Hypothesis is that is different from 30 mins
# You can't assume normality, so you will use a non-parametric test: the Wilcoxon signed rank tes
# The significance level is 0.1
# Read information about the Wilcoxon signed rank test. You will have to:

# Find the difference between every data point from the assumed median
# Find the absolute value of each of those differences and rank them in ascending order
# Compute $s_{+}$ as the sum of the rank of the positive differences
# Determine the critical value. Look for Wilcoxon signed rank table: https://www.real-statistics.com/statistics-tables/wilcoxon-signed-ranks-table/ 
# Determine the upper critical value. This is calculated summing all possible ranks: $$s_{+,max} = 1 + 2 + 3 + ... + n$$
# Can you reject the null hypothesis?
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as ss
import matplotlib.pylab as plt

delivery_data= pd.read_csv("https://aicoreassessments.s3.amazonaws.com/amazon_data.csv")
delivery_data = delivery_data.iloc[: , 1:] # remove first column 
delivery_data.rename(columns={delivery_data.columns[0]: "delivery_time" },inplace=True)
#%%
delivery_data
plt.figure("Delivery Plots")
sns.distplot(delivery_data.delivery_time, label='T')
plt.legend()
plt.show()

#%%
print(delivery_data.describe())
median = delivery_data.delivery_time.median()
delivery_data["diff"] = delivery_data.delivery_time - median
delivery_data.loc[delivery_data["diff"] < 0, "sign"] = "-"
delivery_data.loc[delivery_data["diff"] >= 0, "sign"] = "+"

delivery_data = (
        delivery_data.assign(abs_diff=delivery_data['diff'].abs())
        .sort_values(['abs_diff'],ascending=True)
        .drop('abs_diff', 1)
        .reset_index()
        .rename(columns={"index": "rank" })
        )
delivery_data['rank'] = delivery_data.index + 1

sum_negative_ranks = delivery_data.loc[delivery_data['sign'] == "-", 'rank'].sum()
sum_positive_ranks = delivery_data.loc[delivery_data['sign'] == "+", 'rank'].sum()

print((sum_negative_ranks + sum_positive_ranks) == (17 * 18)/2) # should be true

# Can't reject the null hypothesis

# Under the null hypothesis, we would expect the distribution of the differences to be approximately symmetric around zero and the the distribution of positives and negatives
# to be distributed at random among the ranks. Under this assumption, it is possible to
# 1
# work out the exact probability of every possible outcome for W. To carry out the test,
# we therefore proceed as follows:
# 6. Choose W = min(W−, W+).
# 7. Use tables of critical values for the Wilcoxon signed rank sum test to find the
# probability of observing a value of W or more extreme. Most tables give both one-sided
# and two-sided p-values. If not, double the one-sided p-value to obtain the two-sided
# p-value. This is an exact test





# %%
