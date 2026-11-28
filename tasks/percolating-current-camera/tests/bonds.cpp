// Direct independent bond samples on a square lattice. Count distinct winding clusters.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <vector>

struct Site { int parent, size, x, y, a, b, rank; };
int main(int argc, char** argv) {
    if (argc != 6) return 2;
    int nx=std::atoi(argv[1]), ny=std::atoi(argv[2]), shear=std::atoi(argv[3]);
    int trials=std::atoi(argv[4]), n=nx*ny;
    std::mt19937_64 rng(std::atoi(argv[5]));
    std::vector<Site> g(n);
    std::map<std::array<int,3>,int> histogram;
    auto find = [&](auto&& self,int i)->int {
        if(g[i].parent==i)return i;
        int j=g[i].parent,root=self(self,j);
        g[i].x+=g[j].x;g[i].y+=g[j].y;return g[i].parent=root;
    };
    auto combine = [&](int i,int a,int b,int rank) {
        if(!rank || g[i].rank==2)return;
        if(rank==2 || (g[i].rank && g[i].a*b!=g[i].b*a)){g[i].rank=2;return;}
        g[i].a=a;g[i].b=b;g[i].rank=1;
    };
    auto link=[&](int i,int j,int x,int y) {
        int ri=find(find,i),rj=find(find,j);
        int dx=g[i].x+x-g[j].x,dy=g[i].y+y-g[j].y;
        if(ri==rj){
            if(dx||dy){int b=dy/ny,a=(dx-b*shear)/nx;
                int d=std::gcd(a,b);a/=d;b/=d;
                if(a<0 || (a==0 && b<0)){a=-a;b=-b;}
                combine(ri,a,b,1);
            }
        }else{
            if(g[ri].size<g[rj].size){std::swap(ri,rj);dx=-dx;dy=-dy;}
            g[rj].parent=ri;g[rj].x=dx;g[rj].y=dy;g[ri].size+=g[rj].size;
            combine(ri,g[rj].a,g[rj].b,g[rj].rank);
        }
    };
    for(int j=0;j<trials;++j) {
        for(int i=0;i<n;++i)g[i]={i,1,0,0,0,0,0};
        uint64_t bits=0;
        for(int i=0;i<n;++i) {
            if(i%32==0)bits=rng();
            int x=i%nx, y=i/nx;
            if(bits&1)link(i,y*nx+(x+1)%nx,1,0);
            if(bits&2)link(i,y+1==ny ? (x-shear+nx)%nx : i+nx,0,1);
            bits>>=2;
        }
        int a=0,b=0,k=0;bool cross=false;
        for(int i=0;i<n;++i)if(g[i].parent==i){
            if(g[i].rank==2)cross=true;
            if(g[i].rank==1){a=g[i].a;b=g[i].b;++k;}
        }
        if(cross)histogram[{0,0,-1}]++;else histogram[{a,b,k}]++;
    }
    for(auto& pair:histogram)std::cout<<pair.first[0]<<" "<<pair.first[1]<<" "<<pair.first[2]<<" "<<pair.second<<"\n";
}
