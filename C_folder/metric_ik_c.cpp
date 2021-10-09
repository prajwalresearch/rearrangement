#include <iomanip>
#include <iostream>
#include<chrono>
#include <vector>
#include <math.h>
#include <nlopt.hpp>
#include <torch/torch.h>
#include<ATen/Context.h>
#include <torch/script.h>
#include <eigen3/Eigen/Dense>
#include<iostream>
#include <random>
#include <string>
#include <map>

using namespace std;
using namespace Eigen;
#define pi  3.14159265358979323846;
class Valuefunction{
    private:
    
        torch::jit::script::Module moduleload;


        torch::Tensor LeakyActivationFunction(const torch::Tensor& x)
        {   auto retVal = torch::leaky_relu(x);
        return retVal;
        }

        torch::Tensor SoftplusActivationFunction(const torch::Tensor& x)
        {   auto retVal = torch::softplus(x);
        return retVal;
        }
        torch::Tensor assemble_lower_triangular_matrix(const torch::Tensor& Lo , const torch::Tensor& Ld){
            assert (2*Lo.size(0) == (Ld.size(1) - Ld.size(1)));
            auto diagonal_matrix = torch::diag_embed(Ld);
            auto l = diagonal_matrix.size(0);
            auto b= diagonal_matrix.size(1);
            auto L = torch::tril(torch::ones({l,b}));
            L[L==1]= Lo.view(-1);
            L = L  + diagonal_matrix; 
            return L;
        }
    public:
           Valuefunction(){
            try {
                const char* argv = "/home/prajwal/MetricIKoptim/model_750_model_epoch_20000.pt";
                // Deserialize the ScriptModule from a file using torch::jit::load().
                moduleload = torch::jit::load(argv);
                moduleload.eval();
                //torch::jit::script::Module module;
            }
            catch (const c10::Error& e) {
                std::cerr << "error loading the model\n";
            }

            std::cout << "okk-model-loaded\n";
        }
       vector<vector<double>>  forwardpass(const vector<double>& x){
            vector<float> q(7);
            for(int i =0;i<7;i++){
                q[i] = float(x[i]);
            }
            std::vector<torch::IValue> input;

            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            auto inputTensor = torch::zeros({6}, options);
            inputTensor = torch::from_blob(q.data(), {6}, options);
            input.push_back(inputTensor);
            torch::Tensor out = moduleload.forward(input).toTensor().cpu();
            //cout<<"valuefunction = "<<out<<endl;
            std::vector<double> outvector(out.data_ptr<float>(), out.data_ptr<float>() + out.numel());
            //for (auto a : outvector) std::cout << a << std::endl;
            int k=0;
            vector<vector<double>> Out2dval(3,vector<double>(3));
            for(int i=0;i<3;i++){
                for(int j=0;j<3;j++)
                    {Out2dval[i][j] = outvector[k];
                    k++;}
            }
            return Out2dval;
        }

        vector<float> forwardpass(const torch::Tensor inputTensor){
            std::vector<torch::IValue> input;
            input.push_back(inputTensor);
            torch::Tensor out = moduleload.forward(input).toTensor().cpu();

            std::vector<float> vec(out.data_ptr<float>(), out.data_ptr<float>() + out.numel());
            //for (auto a : vec) std::cout << a << std::endl;


            return vec;
             }

};



typedef struct {

 vector<vector<double> > value;
 vector<double >  goalCartValue;


}my_func_data;

double myfunc(const vector<double> &x , vector<double> &grad, void* fundata){
    my_func_data *d = (my_func_data *) fundata;
    vector<vector<double> > vfxn = d->value;
    vector<double >  goalcart  = d->goalCartValue;
    vector<double > diff;
    vector<double> currentCart(3,0.0) ;
    vector<double> currentqdot(7,0.0);
    for (int i =0 ; i<3;i++){
        currentCart[i] = x[i+7];

    }
    for(int i=0;i<7;i++)
            currentqdot[i] =x[i];
    Eigen::Vector3d goalVector = Eigen::Map<Eigen::VectorXd,Eigen::Unaligned>(goalcart.data(),goalcart.size());
    Eigen::Vector3d CurrentVector = Eigen::Map<Eigen::VectorXd,Eigen::Unaligned>(currentCart.data(),currentCart.size());
    Eigen::VectorXd CurrentqdotVec = Eigen::Map<Eigen::VectorXd,Eigen::Unaligned>(currentqdot.data(),currentqdot.size());
    Eigen::MatrixXd valuematrix(3,3);
    for (int i =0 ; i<3;i++)
        valuematrix.row(i) = VectorXd::Map(&vfxn[i][0],vfxn[i].size());
    Eigen::Vector3d  diffVector = CurrentVector - goalVector;
    double Squaeqnorm = CurrentqdotVec.squaredNorm();
    //Eigen::VectorXd Totalcost = diffVector.transpose()*valuematrix*diffVector ;
    
    double total_cost_double  = diffVector.transpose()*valuematrix*diffVector ;
    //memcpy(&total_cost_double , Totalcost.data() , sizeof(double)*1);
    double w_des_vel  = 0.002;
    total_cost_double = total_cost_double +  w_des_vel*Squaeqnorm;
    Eigen::VectorXd Jacob_cost  = valuematrix.transpose()*diffVector + valuematrix*diffVector;
    
    //cout<<"total_cost_double"<<;
    if(!grad.empty()){    
        //cout<<"inside grad of cost func"<<endl;
        grad[0] = 2*w_des_vel*currentqdot[0];
        grad[1] =  2*w_des_vel*currentqdot[1];
        grad[2] =  2*w_des_vel*currentqdot[2];
        grad[3] =  2*w_des_vel*currentqdot[3];
        grad[4] =  2*w_des_vel*currentqdot[4];
        grad[5] =  2*w_des_vel*currentqdot[5];
        grad[6] =  2*w_des_vel*currentqdot[6];
        grad[7] = Jacob_cost(0) ;
        grad[8] = Jacob_cost(1);
        grad[9] = Jacob_cost(2);
    }

    return  total_cost_double ;
}

typedef struct {

    vector<vector<double> > value;
    vector<double >  goalCartValue;
    vector<double > prevSolverVariable;
    vector<double> q_min;
    vector<double> q_max;
    vector<double> currentq;
    vector<double> qacc_max;
    vector<double> qacc_min;
    double lingradA[14][10];
    double lingradB[14][10];



}my_costfunc_data;

void  linconstA(my_costfunc_data &constCost){
    double t =0.02;
    Matrix <double, 7, 10,RowMajor>  ub;
    ub = Matrix<double, 7, 10,RowMajor>::Identity()*t;
    Matrix <double, 7, 10,RowMajor>  lb;
    lb =Matrix<double, 7, 10,RowMajor>::Identity()*(-t);
    Matrix<double, 14, 10,RowMajor>D;
    D<<ub,
        lb;
    //cout<<D;
    for(int i=0;i<14;i++){
        for(int j=0;j<10;j++){
            constCost.lingradA[i][j] = D(i,j);
        }
    }
    return ;
}

void  linconstB(my_costfunc_data &constCost){
    double t =0.02;
    Matrix <double, 7, 10,RowMajor>  ub;
    ub = Matrix<double, 7, 10,RowMajor>::Identity()*1;
    Matrix <double, 7, 10,RowMajor>  lb;
    lb =Matrix<double, 7, 10,RowMajor>::Identity()*(-1);
    Matrix<double, 14, 10,RowMajor>D;
    D<<ub,
        lb;
    //cout<<D;
    for(int i=0;i<14;i++){
        for(int j=0;j<10;j++){
            constCost.lingradB[i][j] = D(i,j);
        }
    }

}
vector<vector<double>> cost_jac(const vector<double>& SolverVariable , const vector<double>& prevSolverVariable);
void linear_constraintA(unsigned m , double *result , unsigned n , const double *x, double* grad , void* f_data ){
    double t =0.02;
    //n is the lenght of x (decision variable) , m is the dimesnsion of objective function
    my_costfunc_data *d = (my_costfunc_data *) f_data;
    vector<double > currentVariable(10,0.0);
    for (int i=0;i<10;i++){
        currentVariable[i] = x[i];
    }
    //std::copy(currentVariable.begin() , currentVariable.end(),  &x[0] )  ;;
    vector<double > prevVariable  = d->prevSolverVariable;
    //vector<double>prevq = d->prevq;
    vector<double> q_max = d->q_max;
    vector<double> q_min = d->q_min;
    vector<double> qacc_max = d->qacc_max;
    vector<double> qacc_min = d->qacc_min;
    vector<double> currentq = d->currentq;
    int k =0;
    if (grad){
            memcpy(grad, & d->lingradA[0][0] , 14*10*sizeof(d->lingradA[0][0]));
            /*for(int i=0;i<14;i++)
            {
                for(int j=0;j<10;j++)
                {
                    grad[k] = d->lingradA[i][j];
                    k = k+ 1 ;
                }
            }*/
    }
    for(int i=0;i<7;i++){
        result[i] = currentVariable[i]*t+currentq[i]-q_max[i];
         result[i+7] = q_min[i] -currentVariable[i]*t- currentq[i];
    }       


    return;
    }
void linear_constraintB(unsigned m , double *result , unsigned n , const double *x, double* grad , void* f_data ){
    double t =0.02;
    //n is the lenght of x (decision variable) , m is the dimesnsion of objective function
    my_costfunc_data *d = (my_costfunc_data *) f_data;
    vector<double > currentVariable(10,0.0);
    for (int i=0;i<10;i++){
        currentVariable[i] = x[i];
    }
    //std::copy(currentVariable.begin() , currentVariable.end(),  &x[0] )  ;;
    vector<double > prevVariable  = d->prevSolverVariable;
    //vector<double>prevq = d->prevq;
    vector<double> q_max = d->q_max;
    vector<double> q_min = d->q_min;
    vector<double> qacc_max = d->qacc_max;
    vector<double> qacc_min = d->qacc_min;
    vector<double> currentq = d->currentq;
    //vector<vector<double>> jac_constr(3,vector<double>(10));
    //jac_constr = cost_jac(currentVariable,prevVariable);


    int k =0;
    if (grad){
            memcpy(grad, & d->lingradB[0][0] , 14*10*sizeof(d->lingradB[0][0]));

    }
    for(int i=0;i<7;i++){
            result[i] = currentVariable[i]-prevVariable[i]-qacc_max[i]*t;
            result[i+7] = qacc_min[i]*t -  currentVariable[i] + prevVariable[i];
        }
    return;
    }
vector<double> fk_nextfranka(const vector<double>& q );
vector<vector<double>> nonlinearcost_jac(const vector<double>& SolverVariable , const vector<double>& prevSolverVariable);

void Nonlinear_constraint(unsigned m , double *result , unsigned n , const double *x, double* grad , void* f_data)
{   double t =0.02;
    //n is the lenght of x (decision variable) , m is the dimesnsion of objective function
    my_costfunc_data *d = (my_costfunc_data *) f_data;
    vector<double > currentVariable(10,0) ;
    for (int i=0;i<10;i++){
        currentVariable[i] = x[i];
    }
    //std::copy(currentVariable.begin() , currentVariable.end(),  x )  ;
    vector<double > prevVariable  = d->prevSolverVariable;
    vector<double> currentq = d->currentq;
    vector<vector<double>> jac_constr(3,vector<double>(10));
    jac_constr = nonlinearcost_jac(currentVariable,currentq);
    vector<double> nextq(7);
    for(int i= 0; i<7;i++){
        nextq[i] =  currentVariable[i]*t + currentq[i];
    }

    vector<double> next_cartpred = fk_nextfranka(nextq);
    int k =0;
    if (grad){
        for(int i=0;i<3;i++)
        {
            for(int j=0;j<10;j++)
            {
                grad[k] = jac_constr[i][j];
                k+=1;


            }
        }
        
    }
    for(int i=0;i<3;i++){
        result[i] = next_cartpred[i]-currentVariable[i+7];
    }
    return;
}


/*vector<double> get_pos(int  current_pos ){
  std::cout<<euclidean_6_2[0][0];
  return 0;
}*/

double test(const double a, const double b, const int s) {
    double q1;
    // uncomment to use a non-deterministic seed
    std::random_device rd;
    std::mt19937 gen(rd());
    //std::mt19937 gen(1729);

    std::uniform_real_distribution<> distr(a,b);

    //std::cout << "lower bound == " << distr.a() << std::endl;
    //std::cout << "upper bound == " << distr.b() << std::endl;

    // generate the distribution as a histogram
    std::map<double, int> histogram;
    for (int i = 0; i < s; ++i) {
        ++histogram[distr(gen)];
    }

    // print results
    //std::cout << "Distribution for " << s << " samples:" << std::endl;
    int counter = 0;
    for (const auto& elem : histogram) {
             q1 = elem.first;
    }
    return q1;
}

int main(int argc, const char* argv[])
{   Valuefunction valuefxn;
    //setting bounds //
    double t =0.02;
    vector<double> q_min({-2.87979327, -1.74532925, -2.87979327, -2.87979327, -2.87979327,
       -0.01745329, -2.87979327,-HUGE_VAL,-HUGE_VAL,-HUGE_VAL});
    vector<double> q_max({2.87979327, 1.76278254, 2.87979327, 0.01745329, 2.87979327,
       3.7350046 , 2.87979327,HUGE_VAL,HUGE_VAL,HUGE_VAL});
    vector<double>qacc_max({15. ,  7.5, 10. , 12.5, 15. , 20. , 20.});
    vector<double>qacc_min({-15. ,  -7.5, -10. , -12.5, -15. , -20. , -20.});
    vector<double>Solvervar_min({-2.175, -2.175, -2.175, -2.175, -2.61 , -2.61 , -2.61, -HUGE_VAL,-HUGE_VAL,-HUGE_VAL });
    vector<double>Solvervar_max({2.175, 2.175, 2.175, 2.175, 2.61 , 2.61 , 2.61,HUGE_VAL,HUGE_VAL,HUGE_VAL }); 
    int nooftrajct = 0;
    double TOTALTIME =0.0;
    double no_iter_sucess = 0.0;
    double total_trajct = 0.0;
    double average_one_step_planning_time = 0.0;
    for (int i = 0; i < 2000 ; i++)
    {   vector<vector<double>> Trajectory_Vector;    
        vector<vector<double>> Cartesian_Vector;
        vector<double> a;
        //vector<vector<double>> Trajectory_Vector;      
        total_trajct +=1;
        vector<double> currentq;
        for(int j=0;j<7;++j)
        {    
            currentq.push_back(test(q_min[j] - 1e-3, q_max[j]- 1e-3, 1));
        }  
        vector<double> current_cartpos( fk_nextfranka(currentq) );
        vector<double> samplenextq;
        for(int j=0;j<7;++j)
        {    
            samplenextq.push_back(test(q_min[j]- 1e-3, q_max[j]- 1e-3, 1));
        }  
        vector<double> goalpos(fk_nextfranka(samplenextq));

        vector<double> next_q(7,0.0);
        vector<double> qdot(7,0.0);

        Eigen::Vector3d Eig_goalcart = Eigen::Map<Eigen::VectorXd,Eigen::Unaligned>(goalpos.data(),goalpos.size());
        Eigen::Vector3d Eig_Currentcart = Eigen::Map<Eigen::VectorXd,Eigen::Unaligned>(current_cartpos.data(),current_cartpos.size());
        Eigen::VectorXd Eig_currentq = Eigen::Map<Eigen::VectorXd,Eigen::Unaligned>(currentq.data(),currentq.size());
        Eigen::VectorXd Eig_currentVariable;  
        //cout<<"iteration"<<i<<endl;
        //"eigGoalcart"<<Eig_goalcart<<endl<<"Eig_currentCart"<<Eig_Currentcart<<endl;
        for(int i =0 ; i<7;i++)
        {
            next_q[i] = qdot[i]*t + currentq[i];        
        }
        Eigen::VectorXd Eig_nextq = Eigen::Map<Eigen::VectorXd,Eigen::Unaligned>(next_q.data(),next_q.size());
        vector<double> next_cartpos = fk_nextfranka(next_q);
        Eigen::Vector3d Eig_nextCart = Eigen::Map<Eigen::VectorXd,Eigen::Unaligned>(next_cartpos.data(),next_cartpos.size());
        vector<double> ValuefxnInput(6,0.0);
        //fix the goal
        my_func_data dataCost;
        std::copy(goalpos.begin() , goalpos.end() , ValuefxnInput.begin()+3 );
        dataCost.goalCartValue =goalpos; 
        vector<double > currentSolverVariable(10 , 0.0);
        std::copy( next_cartpos.begin() , next_cartpos.end() , currentSolverVariable.begin()+7);
        my_costfunc_data constCost;
        constCost.goalCartValue =goalpos;
        constCost.q_min = q_min;
        constCost.q_max =q_max;
        constCost.qacc_min =qacc_min;
        constCost.qacc_max =qacc_max;

        linconstA(constCost);
        linconstB(constCost);

        auto fptr = &dataCost; 
        void *funcdata_ptr = reinterpret_cast<void *&>(fptr); 
        auto constr_fptr = &constCost; 
        void *costr_ptr = reinterpret_cast<void *&>(constr_fptr);
        double minf;
        vector<double> lintol_constraint(14,1e-3);
        vector<double> nonlintol_constraint(3,1e-6);
        nlopt::opt opt(nlopt::LD_SLSQP,10);
        opt.set_lower_bounds(Solvervar_min);
        opt.set_upper_bounds(Solvervar_max);
        opt.set_min_objective(myfunc , funcdata_ptr);
        opt.add_inequality_mconstraint(linear_constraintA,costr_ptr,lintol_constraint);
        opt.add_inequality_mconstraint(linear_constraintB,costr_ptr,lintol_constraint);
        opt.add_equality_mconstraint(Nonlinear_constraint,costr_ptr ,nonlintol_constraint);
        //opt.set_xtol_rel(1e-05);
        //opt.set_xtol_abs(1e-09);
        opt.set_ftol_rel(1e-3);
        opt.set_ftol_abs(1e-06);
        //opt.set_stopval(1e-04);
        opt.set_maxeval(400);
        int itr=1;
        //cout<<"before loop"<<ensdl;
        double totaltime =0.0;
        double delT;
        double totaljointtrajc =0.0;
        double totalcarttrajc = 0.0;
        int iteration =0;
        double totalCrtresidual = 0.0;
        cout<<"befre norm="<<(Eig_goalcart - Eig_Currentcart).norm()<<endl;
        while((Eig_goalcart - Eig_Currentcart).norm() > 0.06  and iteration <6000)
        {   //cout<<"curent cart="<<Eig_Currentcart<<endl<<"----"<<" eig goal"<<Eig_goalcart;
        
            iteration+=1;
            //cout<<"---itr=----"<<itr<<endl;
            std::copy(current_cartpos.begin() , current_cartpos.end() , ValuefxnInput.begin() );
            vector<vector<double> > ptr= valuefxn.forwardpass(ValuefxnInput);
            dataCost.value = ptr;
            constCost.value = ptr;
            constCost.currentq =currentq;
            constCost.prevSolverVariable = currentSolverVariable;
            //Eig_currentCart = Eigen::Map<Eigen::VectorXd,Eigen::Unaligned>(current_cartpos.data(),current_cartpos.size());
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            try
            {
                nlopt::result result = opt.optimize(currentSolverVariable,minf) ;
                Eig_currentVariable =Eigen::Map<Eigen::VectorXd,Eigen::Unaligned>(currentSolverVariable.data(),currentSolverVariable.size());
                //cout<<"Eig_currentVariable==="<<Eig_currentVariable<<endl;
            }
            catch(std::exception &e) {
            std::cerr << "nlopt failed: " << e.what() << std::endl;
            std::copy(currentSolverVariable.begin()+7 , currentSolverVariable.end() , current_cartpos.begin() );
            
            for(int i=0;i<7;i++)
                next_q[i] = currentSolverVariable[i]*t + currentq[i];
            next_cartpos =  fk_nextfranka(next_q );
            Eig_nextCart = Eigen::Map<Eigen::VectorXd,Eigen::Unaligned>(next_cartpos.data(),next_cartpos.size());
            Eig_nextq = Eigen::Map<Eigen::VectorXd,Eigen::Unaligned>(next_q.data(),next_q.size());        
            //cout<<"next cart pos = "<< Eig_nextCart<<endl<<"next q" << Eig_nextq <<endl;
            return EXIT_FAILURE;
            
            }
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::copy(currentSolverVariable.begin()+7 , currentSolverVariable.end() , current_cartpos.begin() );
            for(int i=0;i<7;i++)
                next_q[i] = currentSolverVariable[i]*t + currentq[i];
            next_cartpos =  fk_nextfranka(next_q );
            delT = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
            //std::cout <<" itreration no= "<< itr << " Time difference = "<<  delT<< "[ms]" << "minf="<<minf<< std::endl;
            //std::cout<<" xyz= "<<Eig_Currentcart<<endl;
            std::copy(next_cartpos.begin() , next_cartpos.end() ,currentSolverVariable.begin()+7 );
            Eig_nextCart = Eigen::Map<Eigen::VectorXd,Eigen::Unaligned>(next_cartpos.data(),next_cartpos.size());
            Eig_nextq = Eigen::Map<Eigen::VectorXd,Eigen::Unaligned>(next_q.data(),next_q.size());        
            totaljointtrajc += (Eig_currentq - Eig_nextq).norm();
            totalcarttrajc += ( Eig_Currentcart- Eig_nextCart).norm();
            Eig_currentq = Eig_nextq;
            Eig_Currentcart = Eig_nextCart;
            currentq = next_q;
            current_cartpos = next_cartpos;
            totaltime+=delT;
            itr+=1;

        }
        /*cout<<"next cart pos = "<< Eig_nextCart<<endl<<"next q" << Eig_nextq <<endl;
        cout<<"total time="<<totaltime<<"ms"<<endl;*/
        //cout<<"total joint space ="<<totaljointtrajc;
        //cout<<"total cart space= "<< totalcarttrajc<<endl;
        //cout<<"TOTAL-TIME = "<<TOTALTIME<<"ms";
        cout<<endl;
        
        if((Eig_goalcart - Eig_Currentcart).norm() <= 0.07 )
        {   no_iter_sucess +=iteration;
            TOTALTIME+=totaltime;
            cout<<"Total time="<< TOTALTIME<<"ms"<<endl;
            cout<<"after norm="<<(Eig_goalcart - Eig_Currentcart).norm()<<endl;
            totalCrtresidual += (Eig_goalcart - Eig_Currentcart).norm(); 
            nooftrajct+=1;
            cout<<"no of sucess trajct = "<< nooftrajct<<endl;
            cout<<"Avg-cart-residual="<< totalCrtresidual/nooftrajct<<endl;
            cout<<"no of total iteration in sucess trajct = "<< no_iter_sucess<<endl;
            average_one_step_planning_time += totaltime/iteration ;
            cout<<"average one step planning time="<< average_one_step_planning_time/nooftrajct<<"ms"<<endl;
            cout<<"total-trajct-tested="<<total_trajct<<endl;
            cout<<"percentage of sucess ="<<(nooftrajct/total_trajct)*100<<endl;

        }

    }
    cout<<"TOTAL-TIME="<<TOTALTIME<<"ms "<<"  Average time = "<<TOTALTIME/nooftrajct;
    return 0;
}







vector<double> fk_nextfranka(const vector<double>& q ){

    vector<double> qinit = q;
    double qinit_1 = qinit[0];
    double qinit_2 = qinit[1];
    double qinit_3 = qinit[2];
    double qinit_4 = qinit[3];
    double qinit_5 = qinit[4];
    double qinit_6 = qinit[5];
    double qinit_7 = qinit[6];

    double  x_e = 0.20999999999999999*(((-std::sin(qinit_1)*std::sin(qinit_3) + std::cos(qinit_1)*std::cos(qinit_2)*std::cos(qinit_3))*std::cos(qinit_4) + std::sin(qinit_2)*std::sin(qinit_4)*std::cos(qinit_1))*std::cos(qinit_5) - (std::sin(qinit_1)*std::cos(qinit_3) + std::sin(qinit_3)*std::cos(qinit_1)*std::cos(qinit_2))*std::sin(qinit_5))*std::sin(qinit_6) + 0.087999999999999995*(((-std::sin(qinit_1)*std::sin(qinit_3) + std::cos(qinit_1)*std::cos(qinit_2)*std::cos(qinit_3))*std::cos(qinit_4) + std::sin(qinit_2)*std::sin(qinit_4)*std::cos(qinit_1))*std::cos(qinit_5) - (std::sin(qinit_1)*std::cos(qinit_3) + std::sin(qinit_3)*std::cos(qinit_1)*std::cos(qinit_2))*std::sin(qinit_5))*std::cos(qinit_6) + 0.087999999999999995*(-(-std::sin(qinit_1)*std::sin(qinit_3) + std::cos(qinit_1)*std::cos(qinit_2)*std::cos(qinit_3))*std::sin(qinit_4) + std::sin(qinit_2)*std::cos(qinit_1)*std::cos(qinit_4))*std::sin(qinit_6) - 0.20999999999999999*(-(-std::sin(qinit_1)*std::sin(qinit_3) + std::cos(qinit_1)*std::cos(qinit_2)*std::cos(qinit_3))*std::sin(qinit_4) + std::sin(qinit_2)*std::cos(qinit_1)*std::cos(qinit_4))*std::cos(qinit_6) - 0.38400000000000001*(-std::sin(qinit_1)*std::sin(qinit_3) + std::cos(qinit_1)*std::cos(qinit_2)*std::cos(qinit_3))*std::sin(qinit_4) - 0.082500000000000004*(-std::sin(qinit_1)*std::sin(qinit_3) + std::cos(qinit_1)*std::cos(qinit_2)*std::cos(qinit_3))*std::cos(qinit_4) - 0.082500000000000004*std::sin(qinit_1)*std::sin(qinit_3) - 0.082500000000000004*std::sin(qinit_2)*std::sin(qinit_4)*std::cos(qinit_1) + 0.38400000000000001*std::sin(qinit_2)*std::cos(qinit_1)*std::cos(qinit_4) + 0.316*std::sin(qinit_2)*std::cos(qinit_1) + 0.082500000000000004*std::cos(qinit_1)*std::cos(qinit_2)*std::cos(qinit_3);
    double y_e = 0.20999999999999999*(((std::sin(qinit_1)*std::cos(qinit_2)*std::cos(qinit_3) + std::sin(qinit_3)*std::cos(qinit_1))*std::cos(qinit_4) + std::sin(qinit_1)*std::sin(qinit_2)*std::sin(qinit_4))*std::cos(qinit_5) - (std::sin(qinit_1)*std::sin(qinit_3)*std::cos(qinit_2) - std::cos(qinit_1)*std::cos(qinit_3))*std::sin(qinit_5))*std::sin(qinit_6) + 0.087999999999999995*(((std::sin(qinit_1)*std::cos(qinit_2)*std::cos(qinit_3) + std::sin(qinit_3)*std::cos(qinit_1))*std::cos(qinit_4) + std::sin(qinit_1)*std::sin(qinit_2)*std::sin(qinit_4))*std::cos(qinit_5) - (std::sin(qinit_1)*std::sin(qinit_3)*std::cos(qinit_2) - std::cos(qinit_1)*std::cos(qinit_3))*std::sin(qinit_5))*std::cos(qinit_6) + 0.087999999999999995*(-(std::sin(qinit_1)*std::cos(qinit_2)*std::cos(qinit_3) + std::sin(qinit_3)*std::cos(qinit_1))*std::sin(qinit_4) + std::sin(qinit_1)*std::sin(qinit_2)*std::cos(qinit_4))*std::sin(qinit_6) - 0.20999999999999999*(-(std::sin(qinit_1)*std::cos(qinit_2)*std::cos(qinit_3) + std::sin(qinit_3)*std::cos(qinit_1))*std::sin(qinit_4) + std::sin(qinit_1)*std::sin(qinit_2)*std::cos(qinit_4))*std::cos(qinit_6) - 0.38400000000000001*(std::sin(qinit_1)*std::cos(qinit_2)*std::cos(qinit_3) + std::sin(qinit_3)*std::cos(qinit_1))*std::sin(qinit_4) - 0.082500000000000004*(std::sin(qinit_1)*std::cos(qinit_2)*std::cos(qinit_3) + std::sin(qinit_3)*std::cos(qinit_1))*std::cos(qinit_4) - 0.082500000000000004*std::sin(qinit_1)*std::sin(qinit_2)*std::sin(qinit_4) + 0.38400000000000001*std::sin(qinit_1)*std::sin(qinit_2)*std::cos(qinit_4) + 0.316*std::sin(qinit_1)*std::sin(qinit_2) + 0.082500000000000004*std::sin(qinit_1)*std::cos(qinit_2)*std::cos(qinit_3) + 0.082500000000000004*std::sin(qinit_3)*std::cos(qinit_1);
    double z_e = 0.20999999999999999*((-std::sin(qinit_2)*std::cos(qinit_3)*std::cos(qinit_4) + std::sin(qinit_4)*std::cos(qinit_2))*std::cos(qinit_5) + std::sin(qinit_2)*std::sin(qinit_3)*std::sin(qinit_5))*std::sin(qinit_6) + 0.087999999999999995*((-std::sin(qinit_2)*std::cos(qinit_3)*std::cos(qinit_4) + std::sin(qinit_4)*std::cos(qinit_2))*std::cos(qinit_5) + std::sin(qinit_2)*std::sin(qinit_3)*std::sin(qinit_5))*std::cos(qinit_6) + 0.087999999999999995*(std::sin(qinit_2)*std::sin(qinit_4)*std::cos(qinit_3) + std::cos(qinit_2)*std::cos(qinit_4))*std::sin(qinit_6) - 0.20999999999999999*(std::sin(qinit_2)*std::sin(qinit_4)*std::cos(qinit_3) + std::cos(qinit_2)*std::cos(qinit_4))*std::cos(qinit_6) + 0.38400000000000001*std::sin(qinit_2)*std::sin(qinit_4)*std::cos(qinit_3) + 0.082500000000000004*std::sin(qinit_2)*std::cos(qinit_3)*std::cos(qinit_4) - 0.082500000000000004*std::sin(qinit_2)*std::cos(qinit_3) - 0.082500000000000004*std::sin(qinit_4)*std::cos(qinit_2) + 0.38400000000000001*std::cos(qinit_2)*std::cos(qinit_4) + 0.316*std::cos(qinit_2) + 0.33000000000000002;
    vector<double> cartCorf({x_e,y_e,z_e});
    return cartCorf;

    }


vector<vector<double>> nonlinearcost_jac(const vector<double>& SolverVariable , const vector<double>& prevq){
    vector<double> qdot = SolverVariable;
    vector<double> qinit =  prevq;
    double qdot_1 = qdot[0];
    double qdot_2 = qdot[1];
    double qdot_3 = qdot[2];
    double qdot_4 = qdot[3];
    double qdot_5 = qdot[4];
    double qdot_6 = qdot[5];
    double qdot_7 = qdot[6];
    
    
    double qinit_1 = qinit[0];
    double qinit_2 = qinit[1];
    double qinit_3 = qinit[2];
    double qinit_4 = qinit[3];
    double qinit_5 = qinit[4];
    double qinit_6 = qinit[5];
    double qinit_7 = qinit[6];

    double t =0.02;
    double const_1_q_1,const_1_q_2,const_1_q_3,const_1_q_4,const_1_q_5,const_1_q_6,const_1_q_7;
    double const_2_q_1,const_2_q_2,const_2_q_3,const_2_q_4,const_2_q_5,const_2_q_6,const_2_q_7;
    double const_3_q_1,const_3_q_2,const_3_q_3,const_3_q_4,const_3_q_5,const_3_q_6,const_3_q_7;
    double const_1_x,const_2_x,const_3_x,const_1_y,const_2_y , const_3_y,const_1_z,const_2_z, const_3_z;
    const_1_q_1 = 0.082500000000000004*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4) - 0.38400000000000001*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4) - 0.316*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2) - 0.082500000000000004*t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) - 0.082500000000000004*t*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1) + (0.087999999999999995*(-t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4) + (-t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) - t*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::cos(qdot_4*t + qinit_4))*std::cos(qdot_5*t + qinit_5) + 0.087999999999999995*(t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_2*t + qinit_2) - t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3))*std::sin(qdot_5*t + qinit_5))*std::cos(qdot_6*t + qinit_6) + (0.20999999999999999*(-t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4) + (-t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) - t*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::cos(qdot_4*t + qinit_4))*std::cos(qdot_5*t + qinit_5) + 0.20999999999999999*(t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_2*t + qinit_2) - t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3))*std::sin(qdot_5*t + qinit_5))*std::sin(qdot_6*t + qinit_6) + (-0.087999999999999995*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4) + 0.087999999999999995*(t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) + t*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::sin(qdot_4*t + qinit_4))*std::sin(qdot_6*t + qinit_6) + (0.20999999999999999*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4) - 0.20999999999999999*(t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) + t*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::sin(qdot_4*t + qinit_4))*std::cos(qdot_6*t + qinit_6) + (0.082500000000000004*t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) + 0.082500000000000004*t*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::cos(qdot_4*t + qinit_4) + (0.38400000000000001*t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) + 0.38400000000000001*t*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::sin(qdot_4*t + qinit_4);
    const_1_q_2 = 0.38400000000000001*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3) + 0.082500000000000004*t*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3)*std::cos(qdot_4*t + qinit_4) - 0.082500000000000004*t*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3) - 0.082500000000000004*t*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2) + 0.38400000000000001*t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4) + 0.316*t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2) + (0.087999999999999995*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_3*t + qinit_3)*std::sin(qdot_5*t + qinit_5)*std::cos(qdot_1*t + qinit_1) + 0.087999999999999995*(-t*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3)*std::cos(qdot_4*t + qinit_4) + t*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2))*std::cos(qdot_5*t + qinit_5))*std::cos(qdot_6*t + qinit_6) + (0.20999999999999999*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_3*t + qinit_3)*std::sin(qdot_5*t + qinit_5)*std::cos(qdot_1*t + qinit_1) + 0.20999999999999999*(-t*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3)*std::cos(qdot_4*t + qinit_4) + t*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2))*std::cos(qdot_5*t + qinit_5))*std::sin(qdot_6*t + qinit_6) + (-0.20999999999999999*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3) - 0.20999999999999999*t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4))*std::cos(qdot_6*t + qinit_6) + (0.087999999999999995*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3) + 0.087999999999999995*t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4))*std::sin(qdot_6*t + qinit_6);
    const_1_q_3 = -0.082500000000000004*t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3) - 0.082500000000000004*t*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2) + (0.087999999999999995*(t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) - t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::sin(qdot_5*t + qinit_5) + 0.087999999999999995*(-t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3) - t*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2))*std::cos(qdot_4*t + qinit_4)*std::cos(qdot_5*t + qinit_5))*std::cos(qdot_6*t + qinit_6) + (0.20999999999999999*(t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) - t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::sin(qdot_5*t + qinit_5) + 0.20999999999999999*(-t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3) - t*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2))*std::cos(qdot_4*t + qinit_4)*std::cos(qdot_5*t + qinit_5))*std::sin(qdot_6*t + qinit_6) + (0.082500000000000004*t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3) + 0.082500000000000004*t*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2))*std::cos(qdot_4*t + qinit_4) + (0.38400000000000001*t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3) + 0.38400000000000001*t*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2))*std::sin(qdot_4*t + qinit_4) + 0.087999999999999995*(t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3) + t*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2))*std::sin(qdot_4*t + qinit_4)*std::sin(qdot_6*t + qinit_6) - 0.20999999999999999*(t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3) + t*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2))*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_6*t + qinit_6);
    const_1_q_4 = -t*(0.082500000000000004*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) - 0.082500000000000004*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::sin(qdot_4*t + qinit_4) + t*(0.38400000000000001*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) - 0.38400000000000001*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::cos(qdot_4*t + qinit_4) - 0.38400000000000001*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_1*t + qinit_1) - 0.082500000000000004*t*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_4*t + qinit_4) + 0.20999999999999999*(-t*(-std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) + std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::sin(qdot_4*t + qinit_4) + t*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_4*t + qinit_4))*std::sin(qdot_6*t + qinit_6)*std::cos(qdot_5*t + qinit_5) + 0.087999999999999995*(-t*(-std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) + std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::sin(qdot_4*t + qinit_4) + t*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_4*t + qinit_4))*std::cos(qdot_5*t + qinit_5)*std::cos(qdot_6*t + qinit_6) + (-0.20999999999999999*t*(std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) - std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::cos(qdot_4*t + qinit_4) + 0.20999999999999999*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_1*t + qinit_1))*std::cos(qdot_6*t + qinit_6) + (0.087999999999999995*t*(std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) - std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::cos(qdot_4*t + qinit_4) - 0.087999999999999995*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_1*t + qinit_1))*std::sin(qdot_6*t + qinit_6);
    const_1_q_5 = (-0.20999999999999999*t*((-std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) + std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::cos(qdot_4*t + qinit_4) + std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_1*t + qinit_1))*std::sin(qdot_5*t + qinit_5) + 0.20999999999999999*t*(-std::sin(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3) - std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2))*std::cos(qdot_5*t + qinit_5))*std::sin(qdot_6*t + qinit_6) + (-0.087999999999999995*t*((-std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) + std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::cos(qdot_4*t + qinit_4) + std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_1*t + qinit_1))*std::sin(qdot_5*t + qinit_5) + 0.087999999999999995*t*(-std::sin(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3) - std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2))*std::cos(qdot_5*t + qinit_5))*std::cos(qdot_6*t + qinit_6);
    const_1_q_6 = -t*(0.087999999999999995*((-std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) + std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::cos(qdot_4*t + qinit_4) + std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_1*t + qinit_1))*std::cos(qdot_5*t + qinit_5) - 0.087999999999999995*(std::sin(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3) + std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2))*std::sin(qdot_5*t + qinit_5))*std::sin(qdot_6*t + qinit_6) + t*(0.20999999999999999*((-std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) + std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::cos(qdot_4*t + qinit_4) + std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_1*t + qinit_1))*std::cos(qdot_5*t + qinit_5) - 0.20999999999999999*(std::sin(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3) + std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2))*std::sin(qdot_5*t + qinit_5))*std::cos(qdot_6*t + qinit_6) + t*(-0.087999999999999995*(-std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) + std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::sin(qdot_4*t + qinit_4) + 0.087999999999999995*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_4*t + qinit_4))*std::cos(qdot_6*t + qinit_6) - t*(0.20999999999999999*(-std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) + std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::sin(qdot_4*t + qinit_4) - 0.20999999999999999*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_4*t + qinit_4))*std::sin(qdot_6*t + qinit_6);
    const_1_q_7 = 0;
    const_2_q_1 = -0.082500000000000004*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) - 0.082500000000000004*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_1*t + qinit_1) + 0.38400000000000001*t*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_4*t + qinit_4) + 0.316*t*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_1*t + qinit_1) + 0.082500000000000004*t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) + (0.087999999999999995*(-t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3) - t*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2))*std::sin(qdot_5*t + qinit_5) + 0.087999999999999995*(t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_1*t + qinit_1) + (-t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) + t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::cos(qdot_4*t + qinit_4))*std::cos(qdot_5*t + qinit_5))*std::cos(qdot_6*t + qinit_6) + (0.20999999999999999*(-t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3) - t*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2))*std::sin(qdot_5*t + qinit_5) + 0.20999999999999999*(t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_1*t + qinit_1) + (-t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) + t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::cos(qdot_4*t + qinit_4))*std::cos(qdot_5*t + qinit_5))*std::sin(qdot_6*t + qinit_6) + (0.082500000000000004*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) - 0.082500000000000004*t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::cos(qdot_4*t + qinit_4) + (0.38400000000000001*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) - 0.38400000000000001*t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::sin(qdot_4*t + qinit_4) + (-0.20999999999999999*t*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_4*t + qinit_4) - 0.20999999999999999*(t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) - t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::sin(qdot_4*t + qinit_4))*std::cos(qdot_6*t + qinit_6) + (0.087999999999999995*t*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_4*t + qinit_4) + 0.087999999999999995*(t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3) - t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::sin(qdot_4*t + qinit_4))*std::sin(qdot_6*t + qinit_6);
    const_2_q_2 = 0.38400000000000001*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_3*t + qinit_3) + 0.082500000000000004*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3)*std::cos(qdot_4*t + qinit_4) - 0.082500000000000004*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) - 0.082500000000000004*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_2*t + qinit_2) + 0.38400000000000001*t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4) + 0.316*t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2) + (0.087999999999999995*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_3*t + qinit_3)*std::sin(qdot_5*t + qinit_5) + 0.087999999999999995*(-t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3)*std::cos(qdot_4*t + qinit_4) + t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_2*t + qinit_2))*std::cos(qdot_5*t + qinit_5))*std::cos(qdot_6*t + qinit_6) + (0.20999999999999999*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_3*t + qinit_3)*std::sin(qdot_5*t + qinit_5) + 0.20999999999999999*(-t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3)*std::cos(qdot_4*t + qinit_4) + t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_2*t + qinit_2))*std::cos(qdot_5*t + qinit_5))*std::sin(qdot_6*t + qinit_6) + (-0.20999999999999999*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_3*t + qinit_3) - 0.20999999999999999*t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4))*std::cos(qdot_6*t + qinit_6) + (0.087999999999999995*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_3*t + qinit_3) + 0.087999999999999995*t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4))*std::sin(qdot_6*t + qinit_6);
    const_2_q_3 = -0.082500000000000004*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_2*t + qinit_2) + 0.082500000000000004*t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3) + (0.087999999999999995*(-t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_2*t + qinit_2) + t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3))*std::cos(qdot_4*t + qinit_4)*std::cos(qdot_5*t + qinit_5) + 0.087999999999999995*(-t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) - t*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::sin(qdot_5*t + qinit_5))*std::cos(qdot_6*t + qinit_6) + (0.20999999999999999*(-t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_2*t + qinit_2) + t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3))*std::cos(qdot_4*t + qinit_4)*std::cos(qdot_5*t + qinit_5) + 0.20999999999999999*(-t*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) - t*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::sin(qdot_5*t + qinit_5))*std::sin(qdot_6*t + qinit_6) + (0.082500000000000004*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_2*t + qinit_2) - 0.082500000000000004*t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3))*std::cos(qdot_4*t + qinit_4) + (0.38400000000000001*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_2*t + qinit_2) - 0.38400000000000001*t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3))*std::sin(qdot_4*t + qinit_4) + 0.087999999999999995*(t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_2*t + qinit_2) - t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3))*std::sin(qdot_4*t + qinit_4)*std::sin(qdot_6*t + qinit_6) - 0.20999999999999999*(t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_2*t + qinit_2) - t*std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3))*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_6*t + qinit_6);
    const_2_q_4 = t*(-0.38400000000000001*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) - 0.38400000000000001*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::cos(qdot_4*t + qinit_4) - t*(-0.082500000000000004*std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) - 0.082500000000000004*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::sin(qdot_4*t + qinit_4) - 0.38400000000000001*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4) - 0.082500000000000004*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4) + (-0.20999999999999999*t*(-std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) - std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::cos(qdot_4*t + qinit_4) + 0.20999999999999999*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4))*std::cos(qdot_6*t + qinit_6) + (0.087999999999999995*t*(-std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) - std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::cos(qdot_4*t + qinit_4) - 0.087999999999999995*t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4))*std::sin(qdot_6*t + qinit_6) + 0.20999999999999999*(-t*(std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) + std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::sin(qdot_4*t + qinit_4) + t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4))*std::sin(qdot_6*t + qinit_6)*std::cos(qdot_5*t + qinit_5) + 0.087999999999999995*(-t*(std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) + std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::sin(qdot_4*t + qinit_4) + t*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4))*std::cos(qdot_5*t + qinit_5)*std::cos(qdot_6*t + qinit_6);
    const_2_q_5 = (-0.20999999999999999*t*((std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) + std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::cos(qdot_4*t + qinit_4) + std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4))*std::sin(qdot_5*t + qinit_5) + 0.20999999999999999*t*(-std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_2*t + qinit_2) + std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3))*std::cos(qdot_5*t + qinit_5))*std::sin(qdot_6*t + qinit_6) + (-0.087999999999999995*t*((std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) + std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::cos(qdot_4*t + qinit_4) + std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4))*std::sin(qdot_5*t + qinit_5) + 0.087999999999999995*t*(-std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_2*t + qinit_2) + std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3))*std::cos(qdot_5*t + qinit_5))*std::cos(qdot_6*t + qinit_6);
    const_2_q_6 = -t*(0.087999999999999995*((std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) + std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::cos(qdot_4*t + qinit_4) + std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4))*std::cos(qdot_5*t + qinit_5) - 0.087999999999999995*(std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_2*t + qinit_2) - std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3))*std::sin(qdot_5*t + qinit_5))*std::sin(qdot_6*t + qinit_6) + t*(0.20999999999999999*((std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) + std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::cos(qdot_4*t + qinit_4) + std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4))*std::cos(qdot_5*t + qinit_5) - 0.20999999999999999*(std::sin(qdot_1*t + qinit_1)*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_2*t + qinit_2) - std::cos(qdot_1*t + qinit_1)*std::cos(qdot_3*t + qinit_3))*std::sin(qdot_5*t + qinit_5))*std::cos(qdot_6*t + qinit_6) + t*(-0.087999999999999995*(std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) + std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::sin(qdot_4*t + qinit_4) + 0.087999999999999995*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4))*std::cos(qdot_6*t + qinit_6) - t*(0.20999999999999999*(std::sin(qdot_1*t + qinit_1)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) + std::sin(qdot_3*t + qinit_3)*std::cos(qdot_1*t + qinit_1))*std::sin(qdot_4*t + qinit_4) - 0.20999999999999999*std::sin(qdot_1*t + qinit_1)*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4))*std::sin(qdot_6*t + qinit_6);
    const_2_q_7 = 0;
    const_3_q_1 = 0;
    const_3_q_2 = 0.082500000000000004*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4) - 0.38400000000000001*t*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4) - 0.316*t*std::sin(qdot_2*t + qinit_2) + 0.38400000000000001*t*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) + 0.082500000000000004*t*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3)*std::cos(qdot_4*t + qinit_4) - 0.082500000000000004*t*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3) + (-0.087999999999999995*t*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4) + 0.087999999999999995*t*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::sin(qdot_6*t + qinit_6) + (0.20999999999999999*t*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4) - 0.20999999999999999*t*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3))*std::cos(qdot_6*t + qinit_6) + (0.087999999999999995*t*std::sin(qdot_3*t + qinit_3)*std::sin(qdot_5*t + qinit_5)*std::cos(qdot_2*t + qinit_2) + 0.087999999999999995*(-t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4) - t*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3)*std::cos(qdot_4*t + qinit_4))*std::cos(qdot_5*t + qinit_5))*std::cos(qdot_6*t + qinit_6) + (0.20999999999999999*t*std::sin(qdot_3*t + qinit_3)*std::sin(qdot_5*t + qinit_5)*std::cos(qdot_2*t + qinit_2) + 0.20999999999999999*(-t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4) - t*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3)*std::cos(qdot_4*t + qinit_4))*std::cos(qdot_5*t + qinit_5))*std::sin(qdot_6*t + qinit_6);
    const_3_q_3 = -0.087999999999999995*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_3*t + qinit_3)*std::sin(qdot_4*t + qinit_4)*std::sin(qdot_6*t + qinit_6) + 0.20999999999999999*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_3*t + qinit_3)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_6*t + qinit_6) - 0.38400000000000001*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_3*t + qinit_3)*std::sin(qdot_4*t + qinit_4) - 0.082500000000000004*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_4*t + qinit_4) + 0.082500000000000004*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_3*t + qinit_3) + (0.087999999999999995*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_4*t + qinit_4)*std::cos(qdot_5*t + qinit_5) + 0.087999999999999995*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_5*t + qinit_5)*std::cos(qdot_3*t + qinit_3))*std::cos(qdot_6*t + qinit_6) + (0.20999999999999999*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_4*t + qinit_4)*std::cos(qdot_5*t + qinit_5) + 0.20999999999999999*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_5*t + qinit_5)*std::cos(qdot_3*t + qinit_3))*std::sin(qdot_6*t + qinit_6);
    const_3_q_4 = -0.082500000000000004*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_3*t + qinit_3) + 0.38400000000000001*t*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3)*std::cos(qdot_4*t + qinit_4) - 0.38400000000000001*t*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_2*t + qinit_2) - 0.082500000000000004*t*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4) + 0.20999999999999999*(t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_3*t + qinit_3) + t*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4))*std::sin(qdot_6*t + qinit_6)*std::cos(qdot_5*t + qinit_5) + 0.087999999999999995*(t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_3*t + qinit_3) + t*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4))*std::cos(qdot_5*t + qinit_5)*std::cos(qdot_6*t + qinit_6) + (-0.20999999999999999*t*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3)*std::cos(qdot_4*t + qinit_4) + 0.20999999999999999*t*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_2*t + qinit_2))*std::cos(qdot_6*t + qinit_6) + (0.087999999999999995*t*std::sin(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3)*std::cos(qdot_4*t + qinit_4) - 0.087999999999999995*t*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_2*t + qinit_2))*std::sin(qdot_6*t + qinit_6);
    const_3_q_5 = (-0.20999999999999999*t*(-std::sin(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3)*std::cos(qdot_4*t + qinit_4) + std::sin(qdot_4*t + qinit_4)*std::cos(qdot_2*t + qinit_2))*std::sin(qdot_5*t + qinit_5) + 0.20999999999999999*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_5*t + qinit_5))*std::sin(qdot_6*t + qinit_6) + (-0.087999999999999995*t*(-std::sin(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3)*std::cos(qdot_4*t + qinit_4) + std::sin(qdot_4*t + qinit_4)*std::cos(qdot_2*t + qinit_2))*std::sin(qdot_5*t + qinit_5) + 0.087999999999999995*t*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_3*t + qinit_3)*std::cos(qdot_5*t + qinit_5))*std::cos(qdot_6*t + qinit_6);
    const_3_q_6 = -t*(0.087999999999999995*(-std::sin(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3)*std::cos(qdot_4*t + qinit_4) + std::sin(qdot_4*t + qinit_4)*std::cos(qdot_2*t + qinit_2))*std::cos(qdot_5*t + qinit_5) + 0.087999999999999995*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_3*t + qinit_3)*std::sin(qdot_5*t + qinit_5))*std::sin(qdot_6*t + qinit_6) + t*(0.20999999999999999*(-std::sin(qdot_2*t + qinit_2)*std::cos(qdot_3*t + qinit_3)*std::cos(qdot_4*t + qinit_4) + std::sin(qdot_4*t + qinit_4)*std::cos(qdot_2*t + qinit_2))*std::cos(qdot_5*t + qinit_5) + 0.20999999999999999*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_3*t + qinit_3)*std::sin(qdot_5*t + qinit_5))*std::cos(qdot_6*t + qinit_6) - t*(-0.20999999999999999*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_3*t + qinit_3) - 0.20999999999999999*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4))*std::sin(qdot_6*t + qinit_6) + t*(0.087999999999999995*std::sin(qdot_2*t + qinit_2)*std::sin(qdot_4*t + qinit_4)*std::cos(qdot_3*t + qinit_3) + 0.087999999999999995*std::cos(qdot_2*t + qinit_2)*std::cos(qdot_4*t + qinit_4))*std::cos(qdot_6*t + qinit_6);
    const_3_q_7 = 0;
    const_1_x = -1;
    const_2_x = 0;
    const_3_x = 0;
    const_1_y = 0;
    const_2_y = -1;
    const_3_y = 0;
    const_1_z = 0;
    const_2_z = 0;
    const_3_z = -1;
    vector<vector<double>> jac_constr(3,vector<double>(10));
    jac_constr = {{const_1_q_1, const_1_q_2, const_1_q_3, const_1_q_4, const_1_q_5, const_1_q_6, const_1_q_7, const_1_x, const_1_y, const_1_z},
                    {const_2_q_1, const_2_q_2, const_2_q_3, const_2_q_4, const_2_q_5, const_2_q_6, const_2_q_7, const_2_x, const_2_y, const_2_z },
                    {const_3_q_1, const_3_q_2, const_3_q_3, const_3_q_4, const_3_q_5, const_3_q_6, const_3_q_7, const_3_x, const_3_y, const_3_z }};  
    
    return jac_constr;

}