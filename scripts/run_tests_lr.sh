for dataset in GuangzhouTraffic EnergyConsumptionFraunhofer ElectricityLoadDiagrams SanFranciscoTraffic MetroInterstateTrafficVolume 
do
    d=1000
    print=50
    for lr in 0.01 0.001 0.0001 0.00001
    do
	(( time python3 main.py --model KalmanHD --dataset "$dataset" --learning_rate "$lr" --dimension_hd "$d" --print_freq "$print") 2>&1 ) | tee KalmanHD_"$dataset";
    done
done

