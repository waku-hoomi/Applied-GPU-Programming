#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover



namespace {

struct MoverMesh {
    const FPfield *XN, *YN, *ZN, *Ex, *Ey, *Ez, *Bxn, *Byn, *Bzn;
    int ny, nz;
    double xStart, yStart, zStart;
    FPfield invdx, invdy, invdz, invVOL;
    double lx, ly, lz;
    bool periodicX, periodicY, periodicZ;

    __device__ size_t index(int x, int y, int z) const {
        return (static_cast<size_t>(x) * ny + y) * nz + z;
    }
};

__device__ void particleBoundary(FPpart& position, FPpart& velocity,
                                 double length, bool periodic)
{
    if (position > length) {
        if (periodic) {
            position -= length;
        } else {
            position = 2 * length - position;
            velocity = -velocity;
        }
    }
    if (position < 0) {
        if (periodic) {
            position += length;
        } else {
            position = -position;
            velocity = -velocity;
        }
    }
}

__global__ void moveParticles(FPpart* state, size_t count, MoverMesh mesh,
                              int subcycles, int iterations, FPpart dt_sub_cycling,
                              FPpart dto2, FPpart qomdt2)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;

    FPpart x = state[i], y = state[count + i], z = state[2 * count + i];
    FPpart u = state[3 * count + i], v = state[4 * count + i], w = state[5 * count + i];

    for (int subcycle = 0; subcycle < subcycles; ++subcycle) {
        FPpart xptilde, yptilde, zptilde;
        FPpart uptilde = u, vptilde = v, wptilde = w;
        FPpart omdtsq, denom, ut, vt, wt, udotb;
        FPfield Exl, Eyl, Ezl, Bxl, Byl, Bzl;
        FPfield weight[2][2][2], xi[2], eta[2], zeta[2];
        int ix, iy, iz;
        xptilde = x;
        yptilde = y;
        zptilde = z;
        // calculate the average velocity iteratively
        for(int innter=0; innter < iterations; innter++){
            // interpolation G-->P
            ix = 2 +  int((x - mesh.xStart)*mesh.invdx);
            iy = 2 +  int((y - mesh.yStart)*mesh.invdy);
            iz = 2 +  int((z - mesh.zStart)*mesh.invdz);

            // calculate weights
            xi[0]   = x - mesh.XN[mesh.index(ix - 1, iy, iz)];
            eta[0]  = y - mesh.YN[mesh.index(ix, iy - 1, iz)];
            zeta[0] = z - mesh.ZN[mesh.index(ix, iy, iz - 1)];
            xi[1]   = mesh.XN[mesh.index(ix, iy, iz)] - x;
            eta[1]  = mesh.YN[mesh.index(ix, iy, iz)] - y;
            zeta[1] = mesh.ZN[mesh.index(ix, iy, iz)] - z;
            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++)
                        weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * mesh.invVOL;

            // set to zero local electric and magnetic field
            Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

            for (int ii=0; ii < 2; ii++)
                for (int jj=0; jj < 2; jj++)
                    for(int kk=0; kk < 2; kk++){
                        Exl += weight[ii][jj][kk]*mesh.Ex[mesh.index(ix- ii, iy -jj, iz- kk )];
                        Eyl += weight[ii][jj][kk]*mesh.Ey[mesh.index(ix- ii, iy -jj, iz- kk )];
                        Ezl += weight[ii][jj][kk]*mesh.Ez[mesh.index(ix- ii, iy -jj, iz -kk )];
                        Bxl += weight[ii][jj][kk]*mesh.Bxn[mesh.index(ix- ii, iy -jj, iz -kk )];
                        Byl += weight[ii][jj][kk]*mesh.Byn[mesh.index(ix- ii, iy -jj, iz -kk )];
                        Bzl += weight[ii][jj][kk]*mesh.Bzn[mesh.index(ix- ii, iy -jj, iz -kk )];
                    }

            // end interpolation
            omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
            denom = 1.0/(1.0 + omdtsq);
            // solve the position equation
            ut= u + qomdt2*Exl;
            vt= v + qomdt2*Eyl;
            wt= w + qomdt2*Ezl;
            udotb = ut*Bxl + vt*Byl + wt*Bzl;
            // solve the velocity equation
            uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
            vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
            wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
            // update position
            x = xptilde + uptilde*dto2;
            y = yptilde + vptilde*dto2;
            z = zptilde + wptilde*dto2;


        } // end of iteration
        // update the final position and velocity
        u= 2.0*uptilde - u;
        v= 2.0*vptilde - v;
        w= 2.0*wptilde - w;
        x = xptilde + uptilde*dt_sub_cycling;
        y = yptilde + vptilde*dt_sub_cycling;
        z = zptilde + wptilde*dt_sub_cycling;
        particleBoundary(x, u, mesh.lx, mesh.periodicX);
        particleBoundary(y, v, mesh.ly, mesh.periodicY);
        particleBoundary(z, w, mesh.lz, mesh.periodicZ);
    }

    state[i] = x;
    state[count + i] = y;
    state[2 * count + i] = z;
    state[3 * count + i] = u;
    state[4 * count + i] = v;
    state[5 * count + i] = w;
}

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;
    ~DeviceBuffer() {
        if (data)
            cudaFree(data);
    }
};

bool cudaOk(cudaError_t status, const char* operation)
{
    if (status == cudaSuccess)
        return true;
    std::cerr << "GPU mover " << operation << " failed: "
              << cudaGetErrorString(status) << std::endl;
    return false;
}

} // namespace

int mover_PC_gpu(particles* part, EMfield* field, grid* grd, parameters* param)
{
    if (part->nop == 0)
        return 0;
    if (part->nop < 0 || part->n_sub_cycles < 1 || part->NiterMover < 1) {
        std::cerr << "Invalid particle count, subcycle count, or iteration count" << std::endl;
        return 1;
    }

    const FPpart dt = static_cast<FPpart>(param->dt) / static_cast<double>(part->n_sub_cycles);
    const FPpart halfDt = .5 * dt;
    const FPpart qomdt2 = part->qom * halfDt / param->c;
    const size_t count = part->nop;
    const size_t nodes = static_cast<size_t>(grd->nxn) * grd->nyn * grd->nzn;
    FPpart* const components[] = {part->x, part->y, part->z, part->u, part->v, part->w};
    const FPfield* const arrays[] = {
        grd->XN_flat, grd->YN_flat, grd->ZN_flat,
        field->Ex_flat, field->Ey_flat, field->Ez_flat,
        field->Bxn_flat, field->Byn_flat, field->Bzn_flat
    };

    DeviceBuffer<FPpart> state;
    DeviceBuffer<FPfield> meshData;
    if (!cudaOk(cudaMalloc(&state.data, 6 * count * sizeof(FPpart)), "particle allocation") ||
        !cudaOk(cudaMalloc(&meshData.data, 9 * nodes * sizeof(FPfield)), "mesh allocation"))
        return 1;

    for (size_t c = 0; c < 6; ++c)
        if (!cudaOk(cudaMemcpy(state.data + c * count, components[c],
                               count * sizeof(FPpart), cudaMemcpyHostToDevice), "particle upload"))
            return 1;
    for (size_t a = 0; a < 9; ++a)
        if (!cudaOk(cudaMemcpy(meshData.data + a * nodes, arrays[a],
                               nodes * sizeof(FPfield), cudaMemcpyHostToDevice), "mesh upload"))
            return 1;

    const FPfield* data = meshData.data;
    const MoverMesh mesh = {
        data, data + nodes, data + 2 * nodes,
        data + 3 * nodes, data + 4 * nodes, data + 5 * nodes,
        data + 6 * nodes, data + 7 * nodes, data + 8 * nodes,
        grd->nyn, grd->nzn, grd->xStart, grd->yStart, grd->zStart,
        grd->invdx, grd->invdy, grd->invdz, grd->invVOL,
        grd->Lx, grd->Ly, grd->Lz, param->PERIODICX, param->PERIODICY, param->PERIODICZ
    };
    const unsigned int threads = 256;
    const unsigned int blocks = (count + threads - 1) / threads;
    moveParticles<<<blocks, threads>>>(state.data, count, mesh, part->n_sub_cycles,
                                       part->NiterMover, dt, halfDt, qomdt2);
    if (!cudaOk(cudaGetLastError(), "kernel launch") ||
        !cudaOk(cudaDeviceSynchronize(), "kernel execution"))
        return 1;

    for (size_t c = 0; c < 6; ++c)
        if (!cudaOk(cudaMemcpy(components[c], state.data + c * count,
                               count * sizeof(FPpart), cudaMemcpyDeviceToHost), "particle download"))
            return 1;
    return 0;
}

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}
