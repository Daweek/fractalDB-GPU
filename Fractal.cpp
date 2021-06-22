/*
 * Fractal.cpp
 *
 *  Created on: May 31, 2021
 *      Author: Edg@r j.
 */
#include "Fractal.hpp"

Fractal::Fractal(Accel*& gpu, int numClass, int numPoints, int numInstances){
   
	// Cleaning main variables
  m_map = NULL;
	m_weights = NULL;
  m_numMaps = 0;
  m_pGPU = gpu;
  m_numPoints = numPoints;
  m_fileCount = 0;
	m_totalWeights = 0;
  m_numClass  = numClass;
	m_numIteration = 0;
	m_numInstances = numInstances;
  
}


void Fractal::paramGenFromFile(int count){

	ostringstream p;
	p<<fixed<<m_numClass;

  stringstream ss;
	ss<<setw(5)<<setfill('0')<<to_string(count);
	string s = ss.str();


	string _path = "data/csv_density0.20_Class"+p.str()+"/" + s + ".csv";
	//cout<<_path<<endl;

	ifstream in(_path);
	vector<vector<double>> fields;

	if (in) {
			string line;

			while (getline(in, line)) {
					stringstream sep(line);
					string field;

					fields.push_back(vector<double>());

					while (getline(sep, field, ',')) {
							fields.back().push_back(stod(field));
					}
			}
	}

#if 0
	cout.setf(ios::scientific);
	cout.precision(18);
	for (auto row : fields) {
		for (auto field : row) {
			cout << field << ' ';
		}
		cout << '\n';
	}
#endif

	// Allocate to 
	float a,b,c,d,e,f,prob;
	int param_size;
	float sum_proba;
	
	a = b = c = d = e = f = prob = sum_proba = 0.0f;

	param_size = fields.size();
	m_numMaps = param_size;
	//std::cout<<"Param_size:"<<param_size<<std::endl;

	if(m_map != NULL)
		free(m_map);

	m_map = (mapping*)malloc(param_size * sizeof(mapping));
	
	//cout<<"Original from file CSV"<<endl;
	for (int i=0;i <param_size;i++){
		
		a = b = c = d = e = f = 0.0f;
		
		//param_rand.print();
		a = fields[i][0];
		b = fields[i][1];
		c = fields[i][2];
		d = fields[i][3];
		e = fields[i][4];
		f = fields[i][5];
		prob = fields[i][6];
	
		m_map[i] = {a, b, c, d, e, f, prob};
		/*
		cout<<m_map[i].a<<","<<
					m_map[i].b<<","<<
					m_map[i].c<<","<<
					m_map[i].d<<","<<
					m_map[i].x<<","<<
					m_map[i].y<<","<<
					m_map[i].p<<endl;
					*/
	}

}

void Fractal::loadWeightsFromCSV(){
	stringstream ss;
		ss<<setw(5)<<setfill('0')<<to_string(m_fileCount);
		string s = ss.str();

		string _path = "data/weights/weights_0.4.csv";
		//cout<<_path<<endl;

		ifstream in(_path);
		vector<vector<double>> fields;

		if (in) {
				string line;

				while (getline(in, line)) {
						stringstream sep(line);
						string field;

						fields.push_back(vector<double>());

						while (getline(sep, field, ',')) {
								fields.back().push_back(stod(field));
						}
				}
		}

	// Allocate to 
	float a,b,c,d,e,f;
	int param_size;
		
	a = b = c = d = e = f = 0.0f;

	param_size = fields.size();
	m_totalWeights = param_size;
	cout<<"Weigths_VectorDim:"<<param_size<<endl;

	if(m_weights != NULL)
		free(m_weights);

	m_weights = (weights*)malloc(param_size * sizeof(weights));
	
	for (int i=0;i <param_size;i++){
		
		a = b = c = d = e = f = 0.0f;
		
		//param_rand.print();
		a = fields[i][0];
		b = fields[i][1];
		c = fields[i][2];
		d = fields[i][3];
		e = fields[i][4];
		f = fields[i][5];
			
		m_weights[i] = {a,b,c,d,e,f};
		/*
		cout<<m_weights[i].wa<<","<<
					m_weights[i].wb<<","<<
					m_weights[i].wc<<","<<
					m_weights[i].wd<<","<<
					m_weights[i].we<<","<<
					m_weights[i].wf<<endl;
					*/
	}

}

void Fractal::paramGenRandom(){

  float a,b,c,d,e,f,prob;
	int param_size;
	float sum_proba;
	
	a = b = c = d = e = f = prob = sum_proba = 0.0f;

	
	random_device                  rand_dev;
  mt19937                        gen(rand_dev());
  uniform_real_distribution<>    dis(-1.0, 1.0);
	uniform_int_distribution<>     distint(2,8);
	
	//param_size = nc::random::randInt<int>(2,8);
	param_size = distint(gen);
	cout<<"Param_size:"<<param_size<<std::endl;
	m_numMaps = param_size;

	m_map = (mapping*)malloc(param_size * sizeof(mapping));

	for (int i=0;i <param_size;i++){
		
		a = b = c = d = e = f = 0.0f;
		auto param_rand = nc::random::uniform<float>({1,6},-1.0f,1.0f);
		//param_rand.print();

		/*
		a = param_rand(0,0);
		b = param_rand(0,1);
		c = param_rand(0,2);
		d = param_rand(0,3);
		e = param_rand(0,4);
		f = param_rand(0,5);
		*/
		a = dis(gen);
		b = dis(gen);
		c = dis(gen);
		d = dis(gen);
		e = dis(gen);
		f = dis(gen);

		prob = abs(a*d - b*c);
		sum_proba += prob;

		m_map[i] = { a, b, c, d, e, f, prob};
		//cout<<m_map[i].x<<endl;
	}

	for (int i=0;i <param_size;i++){
		m_map[i].p /= sum_proba;
		//cout<<m_map[i].p<<endl;
	}

}

void Fractal::appendWeights(int count){

	int param_size = m_numMaps;
	//cout<<"After apply weights:"<<endl;

	for (int i=0;i < param_size;i++){
		
		m_map[i].a *= m_weights[count].wa;
		m_map[i].b *= m_weights[count].wb;
		m_map[i].c *= m_weights[count].wc;
		m_map[i].d *= m_weights[count].wd;
		m_map[i].x *= m_weights[count].we;
		m_map[i].y *= m_weights[count].wf;
		
		/*
		cout<<m_map[i].a<<","<<
					m_map[i].b<<","<<
					m_map[i].c<<","<<
					m_map[i].d<<","<<
					m_map[i].x<<","<<
					m_map[i].y<<endl;
					*/
	}

}


void Fractal::initFractalParam(Settings cfg, int count){

	if(cfg.pa == FROM_CSV){
    paramGenFromFile(count);
    //cout<<"From File"<<endl;
  }

  if(cfg.pa == FROM_RAND){ 
    paramGenRandom();
    //cout<<"From Random"<<endl;
  }

	//std::cout<<std::endl<<std::endl<<std::endl<<std::endl;
}

void Fractal::generateFractal(){

	m_pGPU->fractalKernel(m_numMaps,m_numPoints);

}

Fractal::~Fractal(){

  free(m_map);

}