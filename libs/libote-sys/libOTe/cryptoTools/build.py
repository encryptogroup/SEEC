import os
import platform
import sys
import multiprocessing

def getParallel(args):
    par = multiprocessing.cpu_count()
    for x in args:
        if x.startswith("--par="):
            val = x.split("=",1)[1]
            par = int(val)
            if par < 1:
                par = 1
            idx = args.index(x)
            args[idx] = ""
    return (args,par)


def replace(list, find, replace):
    if find in list:
        idx = list.index(find)
        list[idx] = replace;
    return list

def Build(projectName, argv, install, par, sudo, noConfig):

    osStr = (platform.system())
    buildDir = ""
    config = ""
    buildType = ""
    setup = "--setup" in argv;
    argv = replace(argv, "--setup", "")

    if "--debug" in argv:
        buildType = "Debug"
    else:
        buildType = "Release"
    argv = replace(argv, "--debug", "")


    if osStr == "Windows":
        buildDir = "out/build/x64-{0}".format(buildType)
        config = "--config {0}".format(buildType)
    elif osStr == "Darwin":
        buildDir = "out/build/osx"
    else:
        buildDir = "out/build/linux"

    if not any("DCMAKE_BUILD_TYPE" in s for s in argv):
        argv.append("-DCMAKE_BUILD_TYPE={0}".format(buildType))

    argStr = ""
    for a in argv:
        argStr = argStr + " " + a

    parallel = ""
    if par != 1:
        parallel = " --parallel " + str(par)

    mkDirCmd = "mkdir -p {0}".format(buildDir); 
    CMakeCmd = "cmake -S . -B {0} {1}".format(buildDir, argStr)
    BuildCmd = "cmake --build {0} {1} {2} ".format(buildDir, config, parallel)

    
    InstallCmd = ""
    if sudo:
        sudo = "sudo "
    else:
        sudo = ""


    if install:
        InstallCmd = sudo
        InstallCmd += "cmake --install {0} {1} ".format(buildDir, config)

    
    print("\n\n====== build.py ("+projectName+") ========")
    if not noConfig:
        print(mkDirCmd)
        print(CMakeCmd)

    if not setup:
        print(BuildCmd)
        if len(InstallCmd):
            print(InstallCmd)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n\n")

    if not noConfig:
        os.system(mkDirCmd)
        os.system(CMakeCmd)

    if not setup:
        os.system(BuildCmd)

        if len(sudo) > 0:
            print("installing "+projectName+": {0}\n".format(InstallCmd))

        os.system(InstallCmd)



def help():

    print(" --install \n\tInstructs the script to install whatever is currently being built to the default location.")
    print(" --install=prefix  \n\tinstall to the provided predix.")
    print(" --sudo  \n\twhen installing, use sudo. May require password.")
    print(" --par=n  \n\twhen building do use parallel  builds with n threads. default = num cores.")
    print(" --noauto  \n\twhen building do not automaticly fetch dependancies.")
    print(" --par=n  \n\twhen building do use parallel  builds with n threads. default = num cores.")
    print(" --debug  \n\tdebug build.")
    print("any additioanl arguments are forwared to cmake.\n")

    print("-build the library")
    print("     python build.py")
    print("-build the library with cmake configurations")
    print("     python build.py --debug -DENABLE_SSE=ON")
    print("-build the library and install with sudo")
    print("     python build.py --install --sudo")
    print("-build the library and install to prefix")
    print("     python build.py --install=~/my/install/dir ")



def parseInstallArgs(args):
    prefix = ""
    doInstall = False
    for x in args:
        if x.startswith("--install="):
            prefix = x.split("=",1)[1]
            prefix = os.path.abspath(os.path.expanduser(prefix))
            idx = args.index(x)
            args[idx] = "-DCMAKE_INSTALL_PREFIX=" + prefix
            doInstall = True
        if x == "--install":
            idx = args.index(x)
            osStr = (platform.system())
            if osStr == "Windows":
                args[idx] = "-DCMAKE_INSTALL_PREFIX=c:/lib"
            else:
                args[idx] = "-DCMAKE_INSTALL_PREFIX=/usr/local"
            doInstall = True

    return (args, doInstall)

def main(projectName, argv):

    if "--help" in argv:
        help()
        return 

    sudo = "--sudo" in argv;
    if not sudo:
        argv.append("-DSUDO_FETCH=OFF")

    if "--noauto" in argv:
        argv = replace(argv, "--noauto", "")
        argv.append("-DFETCH_AUTO=OFF")
    else:
        argv.append("-DFETCH_AUTO=ON")

    argv = replace(argv, "--relic", "-DENABLE_RELIC=ON -DFETCH_RELIC=ON")
    argv = replace(argv, "--boost", "-DENABLE_BOOST=ON -DFETCH_BOOST=ON")
    argv = replace(argv, "--sodium", "-DENABLE_SODIUM=ON -DFETCH_SODIUM=ON")
    argv = replace(argv, "--openssl", "-DENABLE_OPENSSL=ON")
    argv = replace(argv, "--sudo", "-DSUDO_FETCH=ON")
        
    argv, install = parseInstallArgs(argv)
    argv, par = getParallel(argv)

    argv.append("-DPARALLEL_FETCH="+str(par))

    noConfig = "--nc" in argv
    argv = replace(argv, "--nc", "")


    Build(projectName, argv, install, par, sudo, noConfig)

if __name__ == "__main__":

    main("cryptoTools", sys.argv[1:])
