# Databricks notebook source
# MAGIC %md
# MAGIC # Definition
# MAGIC - 0_0 -> no booking + no search
# MAGIC - 0_1 -> no booking + act intent (14 days)
# MAGIC - 0_2 -> no booking + act search (6 month)
# MAGIC 
# MAGIC - 1_0 -> with booking + no search
# MAGIC - 1_1 -> with booking + act intent (14 days)
# MAGIC - 1_2 -> with booking + act search (6 month)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC To create features required for DF_0_1 usecase
# MAGIC 
# MAGIC Important Terms
# MAGIC 
# MAGIC 1. Entity
# MAGIC 2. Workspace
# MAGIC 3. Source
# MAGIC 4. Compute
# MAGIC 5. Sink
# MAGIC 
# MAGIC ### Entity 
# MAGIC it is basically a key ( single/composite ) which is used to identify the record for a feature.
# MAGIC 
# MAGIC For example -> uuid_profiletype consists of uuid and profiletype.
# MAGIC 
# MAGIC ```
# MAGIC uuid_profileType = EntityConfig(name = 'user_id', keys = ['uuid', 'profile_type'], description = '')
# MAGIC ```
# MAGIC 
# MAGIC ### Workspace  
# MAGIC Basically a logical group which can also be termed as a project. All the permissions, datasets, features, etc will be created according to the workspace assigned.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Let's create features required for df_0_1 (defined above),to acheive the above, we need three things, source, compute, sink.
# MAGIC 
# MAGIC ### Source
# MAGIC It is a dataset which contains the relevant information of the feature you are trying to create.
# MAGIC 
# MAGIC ### Compute
# MAGIC It means the tranformation sql that you want to apply on source like aggregation.
# MAGIC 
# MAGIC ### Sink
# MAGIC Where the computed dataset actually resides ( mostly s3 location in our case) -- which can be queried using redash also.
# MAGIC 
# MAGIC 
# MAGIC ### Defining a datasource
# MAGIC In our example, we can consider the following as our source dataset.
# MAGIC 
# MAGIC ```
# MAGIC timestampConf = TimestampConfig(col_name = 'timestamp', col_datatype = 'long', col_format = 'epoch_millis')
# MAGIC partitionConf = [PartitionConfig(col_name = 'date_part', is_virtual = False, is_datetime = True, index = 0, datetime_format = 'yyyy-MM-dd')]
# MAGIC s3SourceConf = S3SourceConfig(bucket = '', path = '', format = '', timestamp_conf = timestampConf, partitionConf = partitionConf)
# MAGIC 
# MAGIC batchSource = BatchSource(name = '', s3_config = s3SourceConf)
# MAGIC 
# MAGIC ```
# MAGIC 
# MAGIC -- Basic sanity checks on s3 location on ds creation
# MAGIC 
# MAGIC -- Version control on notebooks vs functions.
# MAGIC 
# MAGIC -- provide feature to make environment available on notebook itself.

# COMMAND ----------

from pyspark.sql import functions as f
from pyspark.sql.types import *
from pyspark.sql import Window
from fs_api.fsclient import *
from fs_api.driver import *
from fs_batch.featurestore_batch import *


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 1. Create Driver

# COMMAND ----------

url = 'http://featurestore.mmt.com/'
api_token = 'API_TOKEN' #optional only needed for registry

driver = FSDriver('url'= url, token = api_token)
#driver.set_credentials

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2. Connect with a workspace

# COMMAND ----------

driver.set_workspace('DS_USER_PERSONALISATION')

// Change sequence

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3. Creating entity

# COMMAND ----------

uuid_profileType = EntityConfig(name = 'user_id', keys = ['uuid', 'profile_type'], description = '')


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4. Define udfs

# COMMAND ----------

import numpy as np
from scipy.stats import entropy

def entropy1(labels):
  if labels:
    labels = np.array(labels)
    value,counts = np.unique(labels, return_counts=True)
    return float(entropy(counts))
  else:
    return None

@udf
def most_common(lst):
  if lst:
    return max(set(lst), key=lst.count)
  else:
    None
    
@udf
def parseDate(date_value):
  if date_value:
    if isinstance(date_value, str):
      if date_value.isdigit():
        parsed_date_value = datetime.datetime.utcfromtimestamp(int(date_value)/1000).strftime('%Y-%m-%d')   
      else:
        parsed_date_value = date_value.split(' ')[0]
    else:
      parsed_date_value = datetime.datetime.utcfromtimestamp(date_value/1000).strftime('%Y-%m-%d')   
    return parsed_date_value
  else:
    return None

def get_lobs(df, lob_col_name, country_col_name):
  
  df = df.withColumn(
    'lob', f.when(
      f.lower(lob_col_name).like('%cab%'), 'cabs').otherwise(
      f.when(
        f.lower(lob_col_name) == 'dom hld', 'dhld').otherwise(
        f.when(
          f.lower(lob_col_name) == 'intl hld', 'ihld').otherwise(
          f.when(
            f.lower(lob_col_name) == 'intl htl', 'ih').otherwise(
            f.when(
               f.lower(lob_col_name) == 'intl flt', 'if').otherwise(
              f.when(
                f.lower(lob_col_name) == 'dom htl', 'dh').otherwise(
                f.when(
                  f.lower(lob_col_name) == 'dom flt', 'df').otherwise(
                  f.when(
                    f.lower(lob_col_name).like('%acme%'), 'acme').otherwise(
                    f.when(f.lower(lob_col_name) == 'rail', 'rails').otherwise(
                      f.when(
                        f.lower(lob_col_name).like('%bus%'), 'bus').otherwise(
                        f.lower(lob_col_name))))))))))))
  
  if country_col_name:
    df = df.withColumn(
      'lob', 
      f.when(
        f.lower(lob_col_name) == 'hol', f.when(
          f.col(country_col_name) != 'IN', 'ihld').otherwise(
          'dhld')).otherwise(
        f.when(
          f.lower(lob_col_name) == 'hotels', f.when(
            f.col(country_col_name) != 'IN', 'ih').otherwise('dh')).otherwise(
          f.when(
            f.lower(lob_col_name) == 'flights', f.when(
              f.col(country_col_name) != 'IN', 'if').otherwise('df')).otherwise(
            f.lower('lob')))))
  
  return df

def get_weight(value, counts_dict, total_count):
  if value is not None and counts_dict is not None and total_count is not None:
    value = str(value)
    if value in counts_dict:
      return round(counts_dict[value]/total_count, 2)
    else:
      return 0.
  else:
    return None
  
def create_map_cities(city_arr):
  count_city_map = {}
  for i, city_tuple in enumerate(city_arr):
    for city in city_tuple:
      if city:
        if city in count_city_map:
          count_city_map[city] += 1
        else:
          count_city_map[city] = 1
  return count_city_map

def create_map(count_arr_of_arr):
  count_map = {}
  for count_arr in count_arr_of_arr:
    count_map[count_arr[0]] = int(count_arr[1])
  return count_map

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5. Define Transformations (derived datasets)

# COMMAND ----------

@transformation(mode = 'pyspark')
def process_users_0_1(df, sector_lob_trends):
  
  lobs = ['df', 'dh', 'if', 'ih', 'bus', 'cabs', 'dhld', 'ihld', 'visa', 'acme', 'metro', 'rails']
  for lob in lobs:
    df = df.withColumn(
      f'if_last_searched_{lob}', f.array_contains(f.col('last_searched_lobs'), lob).cast(LongType()))

  df = df.withColumn(
    'cross_lob_probs', get_cross_lob_probs('last_searched_lobs', 'lob')).withColumn(
    'last_location', 
    f.when(f.col('home_location').isNull(),
           f.when(f.col('last_activity_city_id').isNull(),
                  f.col('last_src_city')).otherwise(
             f.col('last_activity_city_id'))).otherwise(f.col('home_location'))).join(
    sector_lob_trends,
    ['last_location', 'lob', 'date_part'],
    'left').withColumn(
    'sector_to_lob_probs', f.when(f.col('sector_to_lob_probs').isNull(), lob_probs_udf('lob')).otherwise(f.col('sector_to_lob_probs'))).persist()
  
  df = df.withColumn(
    'last_searched_recency', 
    f.when(f.col('last_searched_recency').isNull(), 
           f.when(f.col('last_active_recency').isNull(), 1681).otherwise(
             f.col('last_active_recency'))).otherwise(
      f.col('last_searched_recency'))).withColumn(
    'flavour_device', 
    f.when(f.col('flavour_device').isNull(), 'desktop/pwa').otherwise(
      f.col('flavour_device'))).withColumn(
    'app_age', 
    f.when(f.col('app_age').isNull(), -1).otherwise(
      f.col('app_age'))).persist()
  
  df = df.withColumn(
    'common_pax_adult', f.col('common_pax_adult').cast(IntegerType())).withColumn(
    'common_pax_child', f.col('common_pax_child').cast(IntegerType())).withColumn(
    'common_pax_infant', f.col('common_pax_infant').cast(IntegerType())).withColumn(
#     'if_intl', f.when(f.col('country_code')!='IN', 1).otherwise(0)).withColumn(
    'user_type', f.col('user_type').cast(IntegerType())).withColumn(
    'notification_enabled', f.col('notification_enabled').cast(IntegerType())).persist()
  
  df = df.withColumn(
    'ap', f.when(f.col('ap') < 0, 0).otherwise(f.col('ap'))).withColumn(
    'avg_los', f.when(f.col('avg_los') >= 0, f.col('avg_los'))).fillna(
    0, ['avg_distance', 'avg_los', 'common_pax_adult', 'common_pax_child', 'common_pax_infant', 'ap'])
  
  return df

# COMMAND ----------

@transformation(mode = 'pyspark')
def withHomeLocation(df):
  joinedDf = df.join(df_country_code, ['home_location'],'left')
  .withColumn(
  'home_town', f.when(
    f.col('home_location').isin(['CTBOM', 'CTDEL', 'CTBLR', 'CTHYDERA', 'CTMAA', 'CTCCU', 'CTPNQ', 'CTAMD', 'CTGGN', 'CTXT1']), f.col('home_location')).otherwise(
    f.when(f.col('country_code') != 'IN', f.lit('INTL')).otherwise(f.lit('IN-OTHERS')))).withColumn(
  'home_tier', f.when(
    f.col('home_location').isin(['CTBOM', 'CTDEL', 'CTBLR', 'CTHYDERA', 'CTMAA', 'CTCCU', 'CTPNQ', 'CTAMD', 'CTGGN', 'CTXT1']), f.lit('1')).otherwise(
    f.when(
      f.col('home_location').isin(['CTJAI', 'CTLKO', 'CTPAT', 'CTNOI', 'CTNVM', 'CTGAU', 'CTNAG', 'CTBHO', 'CTBBI', 'CTIXC', 'CTVTZ', 'CTIDR', 'CTGHZ', 'CTSTV', 'CTGOI', 'CTCJB', 'CTBDQ', 'CTCOK', 'CTIXJ', 'CTFARI', 'CTVGA', 'CTIXR', 'CTKNU', 'CTXLD', 'CTSXR', 'CTSHL', 'CTDED', 'CTRPR', 'CTVNS', 'CTIXM', 'CTAGR', 'CTGNOI']), f.lit('2')).otherwise(
      f.when(f.col('country_code') != 'IN', f.lit('INTL')).otherwise(f.lit('3')))))
  
  return joinedDf

# COMMAND ----------

@transformation(mode = 'pyspark')
def user_drop_off_aggregates(user_lob_sc_aggregates, prediction_date):
  df_sc = user_lob_sc_aggregates.withColumn(
    'booking_pageviews', f.coalesce('total_funnel_counts.acme_booking', 'total_funnel_counts.booking', f.lit(0))).withColumn(
#     'details_pageviews', f.coalesce('total_funnel_counts.details', 'total_funnel_counts.detail', f.lit(0))).withColumn(
    'pre_booking_pageviews', f.coalesce('total_funnel_counts.payment', f.lit(0)) + f.coalesce('total_funnel_counts.seatmap', 'total_funnel_counts.ancillary', 'total_funnel_counts.roomSelection',
    f.lit(0))).withColumn('listing_pageviews', f.coalesce('total_funnel_counts.listing', 'total_funnel_counts.collections', 'total_funnel_counts.COLLECTIONS', f.lit(0))).withColumn(
    'review_pageviews', f.coalesce('total_funnel_counts.review', f.lit(0)) + f.coalesce('total_funnel_counts.traveller', f.lit(0))).select(
    'uuid',
    'profile_type',
    f.lower('lob_name').alias('lob'),
    'min_visit_start_time',
    'minimal_sc', 
    'max_funnel_depth',
    'hit_counts', 
    f.when((f.col('total_bookings').isNotNull()) & (f.col('total_bookings') > 0), 1).otherwise(0).alias('if_booked'),
    'max_depth_sc.to_loc.id',
    'max_depth_sc.pax_adult', 
    'max_depth_sc.pax_child',
    'max_depth_sc.from_date', 
    'max_depth_sc.to_date',
    'max_depth_sc.journey_type',
    f.col('max_depth_sc.to_loc.country_code').alias('country_code'),
    'booking_pageviews', 
    'pre_booking_pageviews', 
    'listing_pageviews',
    'review_pageviews')

  df_sc = df_sc.withColumn(
    'lobs_list',
    f.collect_list(
      'lob').over(Window.partitionBy('uuid', 'profile_type'))).withColumn(
    'entropy_lobs', entropy_udf('lobs_list')).withColumn(
    'most_common_lob', most_common('lobs_list'))

  df_sc = df_sc.filter(
    f.lower('lob') == 'df').withColumn(
    'source', f.split('minimal_sc', '::')[0]).withColumn(
    'destination', f.when(
      f.split('minimal_sc', '::')[1] != 'DF', f.split('minimal_sc', '::')[1]).otherwise(f.col('id'))).filter(
    'source != destination')
  
  df_sc = df_sc.withColumn(
    'if_booked_sector', f.max('if_booked').over(
      Window.partitionBy(
        'uuid', 'source', 'destination').orderBy(
        'uuid', f.asc('min_visit_start_time')))).withColumn(
    'if_booked_sc', f.last('if_booked').over(
      Window.partitionBy(
        'uuid', 'source', 'destination', 'pax_adult', 'pax_child', 'from_date', 'to_date', 'journey_type').orderBy(
        'uuid', f.asc('min_visit_start_time')))).groupBy(
    'uuid', 'profile_type', 'source', 'destination', 'pax_adult', 'pax_child', 'from_date', 'to_date', 'journey_type', 'entropy_lobs', 'most_common_lob', 'lob', 'if_booked_sc', 'if_booked_sector').agg(
    f.sum('hit_counts').alias('hit_counts'),
    f.max('max_funnel_depth').alias('max_funnel_depth'),
    f.sum('booking_pageviews').cast(IntegerType()).alias('booking_pageviews'),
    f.sum('pre_booking_pageviews').cast(IntegerType()).alias('pre_booking_pageviews'),
    f.sum('review_pageviews').cast(IntegerType()).alias('review_pageviews'),
    f.sum('listing_pageviews').cast(IntegerType()).alias('listing_pageviews'),
    f.min('min_visit_start_time').alias('min_visit_start_time'))

  df_sc_agg = df_sc.orderBy(
    'uuid', f.asc('min_visit_start_time')).groupBy(
    'uuid', 'profile_type', 'lob', 'entropy_lobs', 'most_common_lob').agg(
    f.collect_list(f.col('if_booked_sc')).alias('if_booked_sc'),
    f.collect_list(f.col('if_booked_sector')).alias('if_booked_sector'),
    f.collect_list(f.col('source')).alias('source'),
    f.collect_list(f.col('destination')).alias('destination'),
    f.collect_list(f.col('pax_adult')).alias('pax_adult'),
    f.collect_list(f.col('pax_child')).alias('pax_child'),
    f.collect_list(f.when(f.col('from_date').isNull(), 0).otherwise(f.col('from_date'))).alias('from_date'),
    f.collect_list(f.when(f.col('to_date').isNull(), 0).otherwise(f.col('to_date'))).alias('to_date'),
    f.collect_list(f.when(f.col('journey_type').isNull(), 'ow').otherwise(f.col('journey_type'))).alias('journey_type'),
    f.collect_list(f.when(f.col('hit_counts').isNull(), 0).otherwise(f.col('hit_counts'))).alias('hit_counts'),
    f.collect_list(f.when(f.col('max_funnel_depth').isNull(), 0).otherwise(f.col('max_funnel_depth'))).alias('max_funnel_depth'),
    f.collect_list('listing_pageviews').alias('listing_pageviews'),
    f.collect_list('review_pageviews').alias('review_pageviews'),
    f.collect_list('pre_booking_pageviews').alias('pre_booking_pageviews'),
    f.collect_list('booking_pageviews').alias('booking_pageviews'),
    f.sum(f.col('listing_pageviews')).alias('total_listing_pageviews'),
    f.sum(f.col('review_pageviews')).alias('total_review_pageviews'),
    f.sum(f.col('pre_booking_pageviews')).alias('total_pre_booking_pageviews'),
    f.sum(f.col('booking_pageviews')).alias('total_booking_pageviews'))

  df_sc_agg = df_sc_agg.withColumn(
    'trips', get_trips_udf(
      'source', 'destination', 'pax_adult', 'pax_child', 'from_date', 'to_date', 'journey_type', 'hit_counts', 'max_funnel_depth', 'if_booked_sc', 'if_booked_sector',
      'listing_pageviews', 'review_pageviews', 'pre_booking_pageviews', 'booking_pageviews')).withColumn(
    'max_hit_trip_active_search', get_max_hit_trip_udf('trips')).withColumn(
    'count_trips_active_search', f.when(f.col('trips')[0][0].isNotNull(), f.size('trips'))).withColumn(
    'trips',  f.explode('trips')).withColumn(
    'destination', f.when(f.col('trips')[1].isNotNull(), f.col('trips')[1])).withColumn(
    'destination_tier', f.when(
      f.col('destination').isNotNull(), f.when(
        f.col('destination').isin(['CTBOM', 'CTDEL', 'CTBLR', 'CTHYDERA', 'CTMAA', 'CTCCU', 'CTPNQ', 'CTAMD', 'CTGGN', 'CTXT1']), f.lit('1')).otherwise(f.when(
          f.col('destination').isin(['CTJAI', 'CTLKO', 'CTPAT', 'CTNOI', 'CTNVM', 'CTGAU', 'CTNAG', 'CTBHO', 'CTBBI', 'CTIXC', 'CTVTZ', 'CTIDR', 'CTGHZ', 'CTSTV', 'CTGOI', 'CTCJB', 'CTBDQ', 'CTCOK', 'CTIXJ', 'CTFARI', 'CTVGA', 'CTIXR', 'CTKNU', 'CTXLD', 'CTSXR', 'CTSHL', 'CTDED', 'CTRPR', 'CTVNS', 'CTIXM', 'CTAGR', 'CTGNOI']), f.lit('2')).otherwise(
          f.lit('3')))))

  df_sc_results = df_sc_agg.groupBy(
    'uuid', 'profile_type', 'lob', 'entropy_lobs', 'most_common_lob', 'max_hit_trip_active_search', 'count_trips_active_search', 'total_listing_pageviews', 'total_review_pageviews', 'total_pre_booking_pageviews', 'total_booking_pageviews').agg(
    most_common(f.collect_list('destination_tier')).cast(IntegerType()).alias('destination_tier_active_search'),
    entropy_udf(f.collect_list(f.col('trips')[0])).alias('entropy_source_active_search'),
    entropy_udf(f.collect_list(f.col('trips')[1])).alias('entropy_destination_active_search'),
    entropy_udf(f.collect_list(f.concat_ws(':', f.col('trips')[4], f.col('trips')[5]))).alias('entropy_dates_active_search'),
    most_common(f.collect_list(f.col('trips')[6])).alias('journey_type_active_search')).withColumn(
    'date_part', f.lit(prediction_date))
  
  return df_sc_results

# COMMAND ----------

@transfomation(mode = 'pyspark')
def enrich_with_search_ap(df_0_1):
  enichedDf = df_0_1.withColumn(
  'search_ap', f.datediff(parseDate(f.col('from_date_active_search')), 'date_part')).withColumn(
  'search_ap_window', 
  f.when(f.col('from_date_active_search') == 0, -1).otherwise( 
    f.when(f.col('search_ap') < 0, -1).otherwise(
      f.when(f.col('search_ap') == 0, 0).otherwise(
        f.when(f.col('search_ap') <= 3, 1).otherwise(
          f.when(f.col('search_ap') <= 7, 2).otherwise(
            f.when(f.col('search_ap') <= 14, 3).otherwise(
              f.when(f.col('search_ap') <= 30, 4).otherwise(
                f.when(f.col('search_ap') <= 60, 5).otherwise(6)))))))))
  return enrichedDf

@transfomation(mode = 'pyspark')
def enrich_with_active_search_cols(df_0_1):
  enichedDf = df_0_1.withColumn(
  'source_active_search', f.col('max_hit_trip_active_search')[0]).withColumn(
  'destination_active_search', f.col('max_hit_trip_active_search')[1]).withColumn(
  'from_date_active_search', f.col('max_hit_trip_active_search')[4].cast(LongType())).withColumn(
  'journey_type_active_search', f.col('max_hit_trip_active_search')[6]).withColumn(
  'hit_counts_active_search', f.col('max_hit_trip_active_search')[7].cast(IntegerType())).withColumn(
  'funnel_depth_active_search', f.col('max_hit_trip_active_search')[8].cast(IntegerType())).withColumn(
  'if_booked_sector_active_search', f.col('max_hit_trip_active_search')[9].cast(IntegerType())).withColumn(
  'listing_pageviews_active_search', f.col('max_hit_trip_active_search')[10].cast(IntegerType())).withColumn(
  'review_pageviews_active_search', f.col('max_hit_trip_active_search')[11].cast(IntegerType())).withColumn(
  'pre_booking_pageviews_active_search', f.col('max_hit_trip_active_search')[12].cast(IntegerType())).withColumn(
  'booking_pageviews_active_search', f.col('max_hit_trip_active_search')[13].cast(IntegerType()))
  return enichedDf

# COMMAND ----------

@transfomation(mode = 'pyspark')
def country_ds(df):
  return df.select(
    f.col('city___locus_city_code___NA___P1D___v1').alias('home_location'), 
    f.col('city___country_code___NA___P1D___v1').alias('country_code')).distinct().persist()
  
@transfomation(mode = 'pyspark')
def users_0_1_ds(df, prediction_date):
  return df.filter('lob ==  "df"').filter(f.col('date_part')==prediction_date)
  
@transformation(mode = 'pyspark')
def user_lob_sc_aggregates_ds(df, prediction_date, search_start_date):
  return df.filter(
      (f.col('date_part') >= search_start_date) & 
      (f.col('date_part') < prediction_date) &
      (f.lower('minimal_sc').like('%::%')) &
      (f.lower('profile_type') == 'personal'))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 6 : Get datasources

# COMMAND ----------

user_lob_sc_aggregates_ds = driver.get_datasource('user_lob_sc_aggregates')
base_location_country_ds = driver.get_datasource_last('country_code_mapper')
users_0_1_ds = driver.get_datasource('users_0_1')
sector_lob_trends = driver.get_datasource('sector_lob_trends')

# COMMAND ----------

# MAGIC %md
# MAGIC ##Step 7. FeatureViews
# MAGIC ###Step 7.1. Creating[Declaring] FeatureViews

# COMMAND ----------

@batch_feature_view(
    name='user_0_1',
    mode='pipeline',
    sources=[user_lob_sc_aggregates_ds, base_location_country_ds, users_0_1_ds, sector_lob_trends]
    entity=uuid_profileType,
    batch_schedule='1d',
    online=True,
    feature_start_time=datetime(2022, 8, 1),
    time_interval=Null,
    tags={'release': 'production'},
    owner='aditya.banerjee@go-mmt.com',
    description='Description',
    context = {'org': 'mmt', 'runDate': prediction_date, 'startDate': search_start_date}
)
// Add a convention for naming this function. and same can be used in feature view name also.

def prepare_df_0_1_data(user_lob_sc_aggregates_ds, base_location_country_ds, users_0_1_ds, sector_lob_trends, context):
  
  #register udfs
  entropy_udf = f.udf(entropy1, DoubleType())
  get_weight_udf = f.udf(get_weight, DoubleType())
  create_map_cities_udf = f.udf(create_map_cities, MapType(StringType(), IntegerType()))
  create_map_udf = f.udf(create_map, MapType(StringType(), IntegerType()))
  
  prediction_date = context['runDate']
  search_start_date = context['startDate']

  base_location_country_ds = country_ds(base_location_country_ds)
  users_0_1_ds = users_0_1_ds(users_0_1_ds, prediction_date)
  user_lob_sc_aggregates_ds = user_lob_sc_aggregates_ds(user_lob_sc_aggregates_ds, prediction_date, search_start_date)
  df_0_1 = withHomeLocation(users_0_1_ds)
  df_0_1 = process_users_0_1(df_0_1, sector_lob_trends)
  df_search = user_drop_off_aggregates(user_lob_sc_aggregates, prediction_date)
  df_0_1 = df_0_1.join(df_search, ['uuid', 'profile_type', 'lob', 'date_part'], 'left') 
  df_0_1 = enrich_with_active_search_cols(df_0_1)
  df_0_1 = enrich_with_search_ap(df_0_1)
    
  select_cols_0_1 = ['lob', 'affluence_level', 'if_uninstalled', 'notification_enabled', 'flavour', 'is_mobile_verified', 'is_email_verified', 'user_type', 'last_searched_recency', 'user_age', 'intent_count', 'intent_score', 'intent_if_coupons', 'day_counts', 'session_counts', 'time_spent', 'hit_counts', 'product_viewed', 'avg_distance', 'avg_los', 'booking_pageviews', 'details_pageviews', 'pre_booking_pageviews', 'listing_pageviews', 'review_pageviews', 'avg_sc_counts', 'common_pax_adult', 'common_pax_child', 'ap', 'last_recency', 'funnel_depth', 'price_bucket', 'cross_lob_probs', 'sector_to_lob_probs', 'app_age', 'avg_session_counts', 'if_last_searched_df', 'if_last_searched_dh', 'count_booking_counts', 'count_booking_counts_mmt', 'user_last_booked_recency', 'has_android', 'has_desktop', 'has_ios', 'has_pwa', 'home_tier', 'home_town', 'avg_solo_tag', 'avg_couple_tag', 'avg_gst_tag', 'loyalty_tier', 'avg_atv_last_3_months', 'count_bookings_last_3_months', 'total_listing_pageviews', 'total_review_pageviews', 'total_pre_booking_pageviews', 'total_booking_pageviews', 'listing_pageviews_active_search', 'review_pageviews_active_search', 'pre_booking_pageviews_active_search', 'search_ap_window', 'entropy_lobs', 'count_trips_active_search', 'destination_tier_active_search', 'entropy_source_active_search', 'entropy_destination_active_search', 'entropy_dates_active_search', 'journey_type_active_search', 'hit_counts_active_search', 'funnel_depth_active_search', 'if_booked_sector_active_search', 'last_booked_lobs', 'last_visit_time', 'user_last_lob_booked_recency', 'max_hit_trip_active_search']

  df_final_0_1 = df_0_1.select(base_identifiers + select_cols_0_1)
  return df_0_1

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7.2. Run FeatureView

# COMMAND ----------

feature_view = driver.get_feature_view("user_0_1")
result_df = feature_view.run(endTime = epoch_ts)  ## compute features but don't persist
result_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7.3. View FeatureView Data from 

# COMMAND ----------

from datetime import datetime, timedelta
start_time = datetime.today() - timedelta(days=2)
result_df = feature_view.get_historical_features(start_time=start_time)
result_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Step 8. Feature Service
# MAGIC ###Step 8.1. Create Feature Service

# COMMAND ----------

buy_propensity_prediction_service = FeatureService(
    name='buy_propensity_prediction_service',
    description='A Feature Service used for supporting a buy propensity prediction model.',
    online_serving_enabled=True,
    features=[
        # add all of the features in a Feature View
        user_0_1,
        # add a single feature from a Feature View using double-bracket notation
        user_0_1[["count"]]
    ],
    tags={'release': 'production'},
    owner="matt@tecton.ai",
    
