for dataset in SanFranciscoTraffic MetroInterstateTrafficVolume GuangzhouTraffic EnergyConsumptionFraunhofer ElectricityLoadDiagrams
do
    if [ "$dataset" = "SanFranciscoTraffic" ]; then
        learning_rate=0.01
        hd_representation=4
        models=8
        for novelty in 0.65 0.75
        do
            python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
        done
        models=32
        for novelty in 0.75 0.8
        do
            python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
        done
        
    elif [ "$dataset" = "MetroInterstateTrafficVolume" ]; then
        learning_rate=0.1
        hd_representation=2
        models=2
        for novelty in 0.2 0.4 0.8
        do
            python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
        done
        models=8
        for novelty in 0.4 0.6
        do
            python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
        done
        models=8
        for novelty in 0.2
        do
            python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
        done
        
    elif [ "$dataset" = "GuangzhouTraffic" ]; then
        learning_rate=0.01
        hd_representation=4
        models=2
        for novelty in 0.65 0.6
        do
            python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
        done
        models=8
        for novelty in 0.65 0.6
        do
            python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
        done
        models=32
        for novelty in 0.65 0.6
        do
            python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
        done

    elif [ "$dataset" = "EnergyConsumptionFraunhofer" ]; then
        learning_rate=0.01
        hd_representation=4
        models=2
        for novelty in 0.25
        do
            python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
        done
        models=8
        for novelty in 0.25
        do
            python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
        done
        models=32
        for novelty in 0.4 0.6
        do
            python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
        done

    elif [ "$dataset" = "ElectricityLoadDiagrams" ]; then
        learning_rate=0.01
        hd_representation=1
        models=2
        for novelty in 0.65 0.75
        do
            python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
        done
        models=8
        for novelty in 0.25 0.35
        do
            python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
        done
        models=32
        for novelty in 0.75 0.8
        do
            python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
        done
    fi
    
    #python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
done