<project xmlns="com.autoesl.autopilot.project" name="pickleball_hls" top="pb_predict">
    <includePaths/>
    <libraryPaths/>
    <Simulation argv="">
        <SimFlow name="csim" ldflags="" mflags="" csimMode="0" lastCsimMode="0"/>
    </Simulation>
    <files xmlns="">
        <file name="../../test_vectors.h" sc="0" tb="1" cflags=" -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="../../tb_pickleball.cpp" sc="0" tb="1" cflags=" -I../.. -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="weights.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="pickleball_model.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="pickleball_model.cpp" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
    </files>
    <solutions xmlns="">
        <solution name="solution1" status="active"/>
    </solutions>
</project>

