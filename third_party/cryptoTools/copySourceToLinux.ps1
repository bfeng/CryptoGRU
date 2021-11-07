$RemoteUserName='rindalp'
$RemoteHostName='eve.eecs.oregonstate.edu'
$PrivateKey='C:\keys\key.ppk'
$SolutionDir=$PWD
$RemoteWorkingDir='/scratch/repo/cryptoTools'

# only files with these extensions will be copied
$FileMasks='**.cpp;**.c;**.h;*.bin,*.S,*.sh,*CMake*;*/gsl/*;*/Tools/*.txt;**.mak;thirdparty/linux/**.get;*/libOTe_Tests/testData/*.txt'

# everything in these folders will be skipped
$ExcludeDirs='.git/;thirdparty/;Debug/;Release/;x64/;ipch/;.vs/'

C:\tools\WinSCP.com  /command `
    "open $RemoteUserName@$RemoteHostName -privatekey=""$PrivateKey"""`
    "call mkdir -p $RemoteWorkingDir"`
    "synchronize Remote $SolutionDir $RemoteWorkingDir -filemask=""$FileMasks|$ExcludeDirs;"" -transfer=binary"`
    "call mkdir -p $RemoteWorkingDir/thirdparty/"`
    "call mkdir -p $RemoteWorkingDir/thirdparty/linux/"`
    "synchronize remote $SolutionDir/thirdparty/linux/ $RemoteWorkingDir/thirdparty/linux/ -filemask=""**.get"" -transfer=binary"`
    "exit" 