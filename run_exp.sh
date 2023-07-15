for dataset in SanFranciscoTraffic MetroInterstateTrafficVolume GuangzhouTraffic EnergyConsumptionFraunhofer ElectricityLoadDiagrams
do
    # python3 main.py --learning_rate 0.00001 --dimension_hd 10000 --model RegHD --dataset "$dataset";

    # python3 main.py --model KalmanFilter --dataset "$dataset";
    
    #for hd_representation in 1 2 4
    #do
        # for learning_rate in 0.1 0.01 0.001 0.0001 0.00001
        # do
        #     python3 main.py --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation"; 
        # done
        
    #    for dimension_hd in 500 1000 2000 5000 10000
    #    do
    #        if [ "$hd_representation" = 1 ]; then
    #        learning_rate=0.00001
    #        elif [ "$hd_representation" = 2 ]; then
    #            learning_rate=0.00001
    #        elif [ "$hd_representation" = 4 ]; then
    #            learning_rate=0.000001
    #        fi

    #        python3 main.py --learning_rate "$learning_rate" --dimension_hd "$dimension_hd" --model RegHD --dataset "$dataset" --hd_representation "$hd_representation";

    #    done    
    #done

    if [ "$dataset" = "SanFranciscoTraffic" ]; then
        learning_rate=0.000001
        hd_representation=4
        dimension_hd=2000
    elif [ "$dataset" = "MetroInterstateTrafficVolume" ]; then
        learning_rate=0.00001
        hd_representation=2
        dimension_hd=1000
    elif [ "$dataset" = "GuangzhouTraffic" ]; then
        learning_rate=0.000001
        hd_representation=4
        dimension_hd=2000
    elif [ "$dataset" = "EnergyConsumptionFraunhofer" ]; then
        learning_rate=0.000001
        hd_representation=4
        dimension_hd=2000
    elif [ "$dataset" = "ElectricityLoadDiagrams" ]; then
        learning_rate=0.00001
        hd_representation=1
        dimension_hd=5000
    fi
    
    #python3 main.py --learning_rate "$learning_rate" --dimension_hd "$dimension_hd" --model RegHD --dataset "$dataset" --hd_representation "$hd_representation";

    python3 main.py --model DNN --dataset "$dataset";

    python3 main.py --model VAE --dataset "$dataset";

done