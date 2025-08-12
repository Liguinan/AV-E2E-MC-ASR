#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -w wav_scp_path -f featdir"
   echo -e "\t-w Description of what is wav_scp_path"
   echo -e "\t-f Description of what is featdir"
   exit 1 # Exit script after printing help
}

while getopts "w:f:" opt
do
   case "$opt" in
      w ) wav_scp_path="$OPTARG" ;;
      f ) featdir="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$featdir" ] || [ -z "$wav_scp_path" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
echo "wav_scp_path=$wav_scp_path"
echo "featdir=$featdir"
