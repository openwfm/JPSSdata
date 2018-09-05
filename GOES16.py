# GOES-16 data from amazon
import boto3
import botocore

s3 = boto3.resource('s3',
         aws_access_key_id=ACCESS_ID,
         aws_secret_access_key= ACCESS_KEY)


bucket1 = s3.Bucket('noaa-goes16')

#Check that bucket exists
try:
    s3.meta.client.head_bucket(Bucket='bucket1')
except botocore.exceptions.ClientError as e:
    # If a client error is thrown, then check that it was a 404 error.
    # If it was a 404 error, then the bucket does not exist.
    error_code = int(e.response['Error']['Code'])
    if error_code == 404:
        exists = False

#Print object key names from bucket        
for bucket in s3.buckets.all():
    for key in bucket.objects.all():
        print(key.key)