for noise in 0.4 # 0.1 0.2 0.5
do
    for dataset in MetroInterstateTrafficVolume GuangzhouTraffic EnergyConsumptionFraunhofer ElectricityLoadDiagrams SanFranciscoTraffic
    do
        if [ "$noise" != "0.5" ] && [ "$noise" != "1" ]; then
            
            python3 main.py --model DNN --dataset "$dataset" --poisson_noise "$noise";
            
            if [ "$dataset" = "SanFranciscoTraffic" ]; then
                learning_rate=0.001
            elif [ "$dataset" = "MetroInterstateTrafficVolume" ]; then
                learning_rate=0.01
            elif [ "$dataset" = "GuangzhouTraffic" ]; then
                learning_rate=0.001
            elif [ "$dataset" = "EnergyConsumptionFraunhofer" ]; then
                learning_rate=0.001
            elif [ "$dataset" = "ElectricityLoadDiagrams" ]; then
                learning_rate=0.001
            fi

            python3 main.py --model KalmanHD --dataset "$dataset" --learning_rate "$learning_rate" --poisson_noise "$noise" --dimension_hd 1000 --device gpu;
            
            python3 main.py --model RegHD --dataset "$dataset" --learning_rate 0.000001 --dimension_hd 10000 --poisson_noise "$noise"  --device cpu;

            python3 main.py --model KalmanFilter --dataset "$dataset" --poisson_noise "$noise";
        fi

        python3 main.py --model VAE --dataset "$dataset" --poisson_noise "$noise";
    done
done
