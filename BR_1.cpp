#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iomanip>
#include <vector>
#include <time.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#define MaxUser 50
#define MaxItem 200
#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
using namespace std;
using namespace Eigen;

struct Graph                                                                            //定义图的结构
{
	string Start[MaxUser];                                                              //始节点，字符串类型
	string End[MaxItem];                                                                //终节点，字符串类型
	double Edge[MaxUser][MaxItem];                                                      //代表图的每条边，及其权值（时间）
};

vector<string> split(string str, char del)                                              //忽略“空位”（::192:168:ABC::416）->（192 168 ABC 416）
{
	stringstream ss(str);
	string tok;
	vector<string> ret;
	while (getline(ss, tok, del))
	{
		if (tok > "")
			ret.push_back(tok);
	}
	return ret;
};

void CreateGraph(Graph *G)                                                              //创建一个图,把数据读入图中
{
	int name_cout = 0;
	string user_name;
	ifstream in_user("user_name.txt");                                                  //把user名读入始节点
	if (in_user) {
		while (getline(in_user, user_name)) {
			G->Start[name_cout] = user_name;
			name_cout++;
			if (name_cout > MaxUser)
				break;
		}
	}
	else
		cout << "no such user_name file" << endl;
	int item_cout = 0;
	string item_name;
	ifstream in_item("item_name.txt");                                                  //把user名读入始节点
	if (in_item) {
		while (getline(in_item, item_name)) {
			G->End[item_cout] = item_name;
			item_cout++;
			if (item_cout > MaxItem)
				break;
		}
	}
	else
		cout << "no such item_name file" << endl;
	for (int i = 0; i < MaxUser; i++)                                                   //初始化图的edge权值，置0
		for (int j = 0; j < MaxItem; j++)
			G->Edge[i][j] = 0;
	string line;
	ifstream in("BA_50_1.txt");                                                         //读取数据中的edge，更改已连接eedge的权值
	if (in) {                                                                           //有该文件
		while (getline(in, line)) {                                                     //line中不包括每行的换行符
			int time;
			string u_name, i_name;
			vector<string> str = split(line, '\t');
			for (int i = 0; i<str.size(); i++) {
				if (i == 0)
					u_name = str[i];
				if (i == 1)
					i_name = str[i];
				if (i == 2)
					time = atoi(str[i].c_str());
			}
			for (int i = 0; i < MaxUser; i++)
				for (int j = 0; j < MaxItem; j++)
					if (strcmp(G->Start[i].c_str(), u_name.c_str()) == 0 && strcmp(G->End[j].c_str(), i_name.c_str()) == 0)
						G->Edge[i][j] = G->Edge[i][j] + pow(0.85, time);                //edge边权值为0.85^time,符合时间效应
		}
		/*
		cout << "权重矩阵G->Edge[i][j]:" << endl;                                       //输出权重矩阵G->Edge[i][j]
		for (int i = 0; i < MaxUser; i++) {
		for (int j = 0; j < MaxItem; j++)
		cout << setprecision(9) << G->Edge[i][j] << " ";
		cout << endl;
		}*/
	}
	else                                                                                // 没有该文件
		cout << "no such file" << endl;
	//system("PAUSE");
}

void Normalize(Graph *G, MatrixXd S, MatrixXd St, double U0[MaxUser], double P0[MaxItem])//对称标准化&迭代
{
	//计算U0
	double delt_U = 0;
	int link_u[MaxUser];
	for (int i = 0; i < MaxUser; i++) {                                                 //计算user的初始MPR值U0
		link_u[i] = 0;
		for (int j = 0; j < MaxItem; j++)
			if (G->Edge[i][j] != 0)
				link_u[i]++;
		delt_U += log(1 + link_u[i]);                                                   //链接数目取对数求和，置为底
	}
	for (int i = 0; i < MaxUser; i++)
		U0[i] = log(link_u[i] + 1) / delt_U;                                                //链接数目取对数/置为底
																						//计算P0
	double delt_I = 0;
	int link_i[MaxItem];
	for (int i = 0; i < MaxItem; i++) {                                                 //计算item的初始MPR值P0
		link_i[i] = 0;
		for (int j = 0; j < MaxUser; j++)
			if (G->Edge[j][i] != 0)
				link_i[i]++;
		delt_I += log(link_i[i]);                                                       //链接数目取对数求和，置为底
	}
	for (int i = 0; i < MaxItem; i++)
		P0[i] = log(link_i[i] + 1) / delt_I;                                                //链接数目取对数/置为底

																						//输出U0、P0的初始值
	cout << "输出U0的初始值：" << endl;
	for (int i = 0; i < MaxUser; i++)
		cout << U0[i] << "\n";
	cout << endl;
	cout << "输出P0的初始值：" << endl;
	for (int i = 0; i < MaxItem; i++)
		cout << P0[i] << "\n";
	cout << endl;
	//计算S=Du*W*Dp
	MatrixXd m;                                                                         //这是一个user*user的对角矩阵，对角线放置user i的链接edge的权值Wij之和
	m.setZero(MaxUser, MaxUser);
	for (int i = 0; i < MaxUser; i++)
		for (int j = 0; j < MaxItem; j++)
			m(i, i) = m(i, i) + G->Edge[i][j];
	MatrixXd n;                                                                         //这是一个item*item的对角矩阵，对角线放置item i的链接edge的权值Wij之和
	n.setZero(MaxItem, MaxItem);
	for (int i = 0; i < MaxItem; i++)
		for (int j = 0; j < MaxUser; j++)
			n(i, i) = n(i, i) + G->Edge[j][i];
	//对矩阵m和n取pow(m,-1/2),pow(n,-1/2)
	MatrixXd di;
	di.setZero(MaxUser, MaxUser);
	for (int i = 0; i < MaxUser; i++)
		di(i, i) = pow(m(i, i), -0.5);
	//std::cout<< "di" << endl << di << endl << endl;
	MatrixXd dj;
	dj.setZero(MaxItem, MaxItem);
	for (int i = 0; i < MaxItem; i++)
		dj(i, i) = pow(n(i, i), -0.5);
	//std::cout << "dj" << endl << dj << endl << endl;
	//cout << "di,dj输出完毕" << endl;
	//把图G的各edge权值转成矩阵W
	MatrixXd W(MaxUser, MaxItem);
	W.setZero(MaxUser, MaxItem);
	for (int i = 0; i < MaxUser; i++)
		for (int j = 0; j < MaxItem; j++)
			W(i, j) = G->Edge[i][j];
	//std::cout<< "W" << endl << W << endl << endl;
	S = di*W*dj;                                                                        //计算S=Du*W*Dp
																						//std::cout << "S" << endl << S << endl << endl;
	St = S.transpose();                                                                 //St为S的转置
																						//std::cout << "St" << endl << St << endl << endl;
																						//cout << "W,S,St输出完毕" << endl;
	cout << "正在进行进行迭代..." << endl;
	VectorXd U(MaxUser), U1(MaxUser), Utemp(MaxUser);
	VectorXd P(MaxItem), P1(MaxItem), Ptemp(MaxItem);
	for (int i = 0; i < MaxUser; i++) {
		Utemp(i) = U0[i];
		U1(i) = U0[i];
	}
	for (int i = 0; i < MaxItem; i++) {
		Ptemp(i) = P0[i];
		P1(i) = P0[i];
	}
	double alpha = 0.85;
	double belt = 0.85;
	for (int i = 0; ; i++) {
		P = Ptemp;
		U = Utemp;
		Ptemp = alpha*(St*U) + (1 - alpha)*P1;
		Utemp = belt*(S*P) + (1 - belt)*U1;
		if (abs(P.sum() - Ptemp.sum()) + abs(U.sum() - Utemp.sum())<1e-15) {            //迭代终止条件：abs<1e-15
			cout << "迭代次数I:" << i << endl;
			break;
		}
	}
	cout << "PR_U_sum:" << U.sum() << endl << "PR_U_sum:" << P.sum() << endl;
	cout << "正在输出各个user的BR值..." << endl;
	cout << setprecision(20) << U << endl;                                              //迭代结束后各个user的BR值
	cout << "正在输出各个item的BR值..." << endl;
	cout << setprecision(20) << P << endl;                                              //迭代结束后各个item的BR值
	system("PAUSE");
}

int main()
{
	MatrixXd S;
	MatrixXd St;
	double U0[MaxUser] = { 0 };
	double P0[MaxItem] = { 0 };
	Graph G;
	CreateGraph(&G);
	Normalize(&G, S, St, U0, P0);
	return 0;
}