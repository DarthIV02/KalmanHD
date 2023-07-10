for dataset in SanFranciscoTraffic MetroInterstateTrafficVolume GuangzhouTraffic EnergyConsumptionFraunhofer ElectricityLoadDiagrams
do
    # python3 main.py --learning_rate 0.00001 --dimension_hd 10000 --model RegHD --dataset "$dataset";

    # python3 main.py --model KalmanFilter --dataset "$dataset";
    
    for hd_representation in 1 2 4 8
    do
        # for learning_rate in 0.1 0.01 0.001 0.0001 0.00001
        # do
        #     python3 main.py --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation"; 
        # done

        python3 main.py --learning_rate 0.00001 --dimension_hd 10000 --model RegHD --dataset "$dataset" --hd_representation "$hd_representation";
    done
done