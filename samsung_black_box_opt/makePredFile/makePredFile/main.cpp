#include <bits/stdc++.h>
#define testSz 4986
using namespace std;

string s;
float curPred;
int srcIdx, dstIdx, mod, cnt, ans[5010];

int main()
{
	ios_base::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	freopen("tmp_pred.txt", "r", stdin);

	for (int i = 0; i < 24855210; i++) {
		cnt = i;
		cin >> s;
		curPred = stof(s);
		srcIdx = cnt / (testSz - 1);
		mod = cnt % (testSz - 1);
		dstIdx = (mod >= srcIdx) ? mod : mod + 1;

		if (curPred >= 0.5) ans[srcIdx]++;
		else ans[dstIdx]++;

		if (cnt % 10000 == 0) cout << cnt << "\n";
		cnt++;
	}

	ofstream output("real_submission.csv");
	output << "ID,y" << "\n";
	for (int i = 0; i < testSz; i++) {
		output << "TEST_"; 
		int curLen = (int)to_string(i).length();
		for (int j = 0; j < 4 - curLen; j++) output << "0";
		output << i << "," << ans[i] << "\n";
	}

	return 0;
}
