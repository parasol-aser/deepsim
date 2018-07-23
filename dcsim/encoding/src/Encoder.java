/**
 * Created by zg on 4/19/17.
 */
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.BitSet;

import com.ibm.wala.classLoader.*;
import org.apache.commons.io.FilenameUtils;

import com.ibm.wala.analysis.typeInference.TypeAbstraction;
import com.ibm.wala.analysis.typeInference.TypeInference;
import com.ibm.wala.core.tests.callGraph.CallGraphTestUtil;
import com.ibm.wala.ipa.callgraph.AnalysisCache;
import com.ibm.wala.ipa.callgraph.AnalysisOptions;
import com.ibm.wala.ipa.callgraph.AnalysisScope;
import com.ibm.wala.ipa.callgraph.impl.Everywhere;
import com.ibm.wala.ipa.cha.ClassHierarchy;
import com.ibm.wala.shrikeBT.IComparisonInstruction;
import com.ibm.wala.shrikeBT.IConditionalBranchInstruction;
import com.ibm.wala.shrikeCT.InvalidClassFileException;
import com.ibm.wala.ssa.*;
import com.ibm.wala.types.ClassLoaderReference;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.util.WalaException;
import com.ibm.wala.util.config.AnalysisScopeReader;
import com.ibm.wala.util.io.FileProvider;
import com.ibm.wala.util.strings.Atom;

public class Encoder {

    final static String exclusionsFileName = "DefaultExclusions.txt";

    // the mSize of the matrix is fixed to 128
    final static int fixedSize = 128;

    // Unprocessed instructions (not included in the current version)
    static String excludedInstruction = "";


    static String base_dir = "data/";

    static String appJar = ""; // the jar file to be analyzed


    public enum BlockType {
        NORMAL,
        LOOP,
        LOOP_BODY,
        IF,
        IF_BODY,
        SWITCH,
        SWITCH_BODY
    }

    static Map<BlockType, Integer> blockWeights = new HashMap<>(BlockType.values().length);

    static {
        blockWeights.put(BlockType.NORMAL, 81);
        blockWeights.put(BlockType.IF, 82);
        blockWeights.put(BlockType.IF_BODY, 83);
        blockWeights.put(BlockType.LOOP, 84);
        blockWeights.put(BlockType.LOOP_BODY, 85);
        blockWeights.put(BlockType.SWITCH, 86);
        blockWeights.put(BlockType.SWITCH_BODY, 87);
    }


    public static void main(String[] args) {
        if (args.length == 0 || !args[0].endsWith(".jar"))
            return;
        appJar = args[0];
        File dataDir = new File(base_dir);
        if (!dataDir.exists())
            dataDir.mkdir();

        try {
            AnalysisScope scope = AnalysisScopeReader.makeJavaBinaryAnalysisScope(appJar,
                    (new FileProvider()).getFile(CallGraphTestUtil.REGRESSION_EXCLUSIONS));

            ClassHierarchy cha = ClassHierarchy.make(scope);

            AnalysisCache cache = new AnalysisCache();
            IRFactory<IMethod> irFactory = cache.getIRFactory();
            int count = 0;
            String jarName = FilenameUtils.removeExtension(FilenameUtils.getName(appJar));
            FileWriter srcFilePathWriter = new FileWriter(base_dir + jarName + ".txt");
            for (IClass cl : cha) {
                //ignore interface or innerclass
                if (cl.isInterface())
                    continue;
                ShrikeClass scl = (ShrikeClass) cl;
                if (scl != null && (scl.isInnerClass() || scl.isStaticInnerClass()))
                    continue;

                String className = cl.getName().toString().replace('/', '.');

                if (!cl.getClassLoader().getReference().equals(ClassLoaderReference.Application))
                    continue;
                for (IMethod m : cl.getDeclaredMethods()) {
                    if (m.isInit())
                        continue;
                    if (m.isClinit())
                        continue;
                    if (m.getSignature().indexOf("$SWITCH_TABLE$") != -1)
                        continue;
                    //ignore abstract method since it ir is null
                    if (m.isAbstract())
                        continue;
                    String mName = m.getName().toString();

                    System.out.println("Method signature: " + m.getSignature());
//					String sig = m.getSignature() + "\n";

                    AnalysisOptions opt = new AnalysisOptions();
                    opt.getSSAOptions().setPiNodePolicy(SSAOptions.getAllBuiltInPiNodes());
                    if (irFactory == null) {
                        System.out.println("irFactory is null.");
                        continue;
                    }
                    IR ir = null;
                    try{
                        ir = irFactory.makeIR(m, Everywhere.EVERYWHERE, opt.getSSAOptions());
                    }
                    catch (java.lang.NullPointerException e){
                        e.printStackTrace();
                        continue;
                    }
                    if (ir == null) {
                        System.out.println("ir is null");
                        continue;
                    }

                    //igonore methods with less than 15 instructions
                    if (ir.getInstructions().length < 15)
                        continue;
                    IBytecodeMethod ibm = (IBytecodeMethod) m;
                    if (m != null) {
                        int bcStrartIndex = ibm.getBytecodeIndex(0);
                        int bcEndIndex = ibm.getBytecodeIndex(ir.getInstructions().length - 1);
                        int srcStartLine = m.getLineNumber(bcStrartIndex);
                        int srcEndLine = m.getLineNumber(bcEndIndex);
                        int loc = srcEndLine = srcEndLine - srcStartLine;
                        if (loc < 5)    //ignroe methods less than 5 lines
                            continue;
                    }

                    computeAdjacencyMatrix(ir, cha, m.getSignature(), className.substring(1));

                    count++;

                    System.out.print(excludedInstruction);
                    excludedInstruction = "";

//					java.lang.System.out.println("**************************");
                }
            }
            srcFilePathWriter.close();
            System.out.println("Data files are in: deepcode-master/data");
            System.out.println("Totally " + Integer.toString(count) + " methods processed.");

        } catch (WalaException | IOException e) {
            e.printStackTrace();
        } catch (InvalidClassFileException e) {
            e.printStackTrace();
        }
    }

    /**
     * Simply detect if this ir contains any invocation, if it does, return true, otherwise false
     *
     * @param ir
     * @return
     */
    public static boolean containInvocation(IR ir, String mName) {
        SSAInstruction[] insList = ir.getInstructions();
        for (SSAInstruction ins : insList) {
            if (ins instanceof SSAInvokeInstruction) {
                String invokeStr = ins.toString().toLowerCase();
                if (invokeStr.contains("compareto") || invokeStr.contains("equals") || invokeStr.contains(mName.toLowerCase()))
                    continue;
                return true;
            }
        }
        return false;
    }

    public static void getBlocksType(SSACFG cfg, BlockType[] types) {
        Iterator<ISSABasicBlock> blockIter = cfg.iterator();
        while (blockIter.hasNext()) {
            ISSABasicBlock block = blockIter.next();
            int blockNumber = block.getNumber();
            Iterator<ISSABasicBlock> succNodes = cfg.getSuccNodes(block);
            if (!block.iterator().hasNext())
                continue;
            SSAInstruction instruction = block.getLastInstruction();
            if (instruction instanceof SSASwitchInstruction) {   //Switch
                types[blockNumber] = BlockType.SWITCH;
                while (succNodes.hasNext()) {
                    ISSABasicBlock succblock = succNodes.next();
                    int succNumber = succblock.getNumber();
                    types[succNumber] = BlockType.SWITCH_BODY;
                }
            }

            if (instruction instanceof SSAConditionalBranchInstruction) {
                types[blockNumber] = BlockType.IF;
                ISSABasicBlock endBlock = null;
                while (succNodes.hasNext()) {
                    endBlock = succNodes.next();
                }
                succNodes = cfg.getSuccNodes(block);
                if (cfg.getSuccNodes(endBlock).next().equals(block)) {   //Loop
                    types[blockNumber] = BlockType.LOOP;
                    while (succNodes.hasNext()) {
                        ISSABasicBlock succblock = succNodes.next();
                        int succNumber = succblock.getNumber();
                        types[succNumber] = BlockType.NORMAL;
                    }
                    int endNumber = endBlock.getNumber();
                    types[endNumber] = BlockType.LOOP_BODY;
                } else {   //IF
                    ISSABasicBlock firstBlock = succNodes.next();
                    int firstNumber = firstBlock.getNumber();
                    types[firstNumber] = BlockType.NORMAL;
                    while (succNodes.hasNext()) {
                        ISSABasicBlock succblock = succNodes.next();
                        int succNumber = succblock.getNumber();
                        types[succNumber] = BlockType.IF_BODY;
                    }
                }
            }
        }
    }

    /**
     * @param ir
     * @param cha
     * @param mName
     * @param className
     */
    public static void computeAdjacencyMatrix(IR ir, ClassHierarchy cha, String mName, String className) {
        SymbolTable st = ir.getSymbolTable();
        int varCount = st.getMaxValueNumber();
        int blockCount = ir.getControlFlowGraph().getNumberOfNodes();

        // class fields of static variables
        List<FieldReference> frList = CodeScanner.getFieldsRead(ir.getInstructions());
        frList.addAll(CodeScanner.getFieldsWritten(ir.getInstructions()));
        // <name of static variable, its number>
        Map<String, Integer> fieldMap = new HashMap<String, Integer>();
        for (FieldReference fr : frList) {
            if (fieldMap.get(fr.getName().toString()) == null)
                fieldMap.put(fr.getName().toString(), ++varCount);
        }

        // indicate the types of all variables
        TypeName[] typeNames = new TypeName[varCount + 1];

        /**
         * typeCode (binary)
         * last two digits:
         * 		local(00), constant(01) or static(10)
         * two digits in the middle:
         * 		common(00), array(01), pointer(10) or reference(11)
         * first four digits:
         * 		9 primitive types, other primitive types, Java library class and user-defined class
         * e.g., 0110 10 01
         *
         * new:
         *
         *
         *
         */
        int[] typeCode = new int[varCount + 1];
        for (int i = 0; i < varCount + 1; i++)
            typeCode[i] = 0; // default value

        TypeInference ti = TypeInference.make(ir, true);
        TypeAbstraction[] taArray = ti.extractAllResults(); //get types for all variable
        for (int i = 0; i < taArray.length; i++) {
            if (taArray[i] != null) {
                if (taArray[i].getTypeReference() != null) {
                    TypeName type = taArray[i].getTypeReference().getName();
                    typeNames[i] = type;

                    if (st.isConstant(i))
                        typeCode[i] = 1;
                    else
                        typeCode[i] = 0;
                }
            }
        }

        for (FieldReference fr : frList) { // for static variable
            int index = fieldMap.get(fr.getName().toString());
            typeNames[index] = fr.getFieldType().getName();
            typeCode[index] = 2;
        }

        for (int i = 0; i < varCount + 1; i++) {
            if (typeNames[i] == null)
                continue;

            // array, pointer, reference or common?
            if (typeNames[i].toString().startsWith("["))
                typeCode[i] = typeCode[i] + 4;
            else if (typeNames[i].toString().startsWith("*"))
                typeCode[i] = typeCode[i] + 8;
            else if (typeNames[i].toString().startsWith("&"))
                typeCode[i] = typeCode[i] + 12;
            else
                typeCode[i] = typeCode[i] + 0;

            // get the innermost type of a variable (without array or reference)
            Atom inner = typeNames[i].getClassName();
            if (inner == Atom.findOrCreateUnicodeAtom("Z"))
                typeCode[i] = typeCode[i] + 16;
            else if (inner == Atom.findOrCreateUnicodeAtom("B"))
                typeCode[i] = typeCode[i] + 32;
            else if (inner == Atom.findOrCreateUnicodeAtom("C"))
                typeCode[i] = typeCode[i] + 48;
            else if (inner == Atom.findOrCreateUnicodeAtom("D"))
                typeCode[i] = typeCode[i] + 64;
            else if (inner == Atom.findOrCreateUnicodeAtom("F"))
                typeCode[i] = typeCode[i] + 80;
            else if (inner == Atom.findOrCreateUnicodeAtom("I"))
                typeCode[i] = typeCode[i] + 96;
            else if (inner == Atom.findOrCreateUnicodeAtom("J"))
                typeCode[i] = typeCode[i] + 112;
            else if (inner == Atom.findOrCreateUnicodeAtom("S"))
                typeCode[i] = typeCode[i] + 128;
            else if (inner == Atom.findOrCreateUnicodeAtom("V"))
                typeCode[i] = typeCode[i] + 144;
            else if (typeNames[i].isPrimitiveType())
                typeCode[i] = typeCode[i] + 160;
            else {
                assert (inner.toString().startsWith("L"));
                if (typeNames[i].getPackage() != null)
                    if (typeNames[i].getPackage().toString().startsWith("java"))
                        typeCode[i] = typeCode[i] + 176;
                    else
                        typeCode[i] = typeCode[i] + 192;
                else
                    typeCode[i] = typeCode[i] + 192;
            }

        }

        /**
         * Adjacency Matrix
         * the computation of the value:
         * 		6-digit opcode + 8-digit "from" type code + 8-digit "to" type code
         * e.g., 001010 01101001 00111000
         *
         * variable-block relationship: 1<<22 (in CFG)
         * block-block relationship: 1<<23 (in CFG)
         *
         * index allocation: [0, varCount+z]
         * local variables 			V1~Vx->[0, x-1]
         * static variables 		[x, varCount-1]
         * basic blocks 			B0~Bz->[varCount, varCount+z]
         */
        int mSize = Math.max(varCount + blockCount, fixedSize);
        // int[][] mat = new int[varCount + blockCount][varCount + blockCount];
//        int[][] mat = new int[mSize][mSize];
        BitSet[][] mat = new BitSet[mSize][mSize];
        int maxVecDim = 89;
        for (int i = 0; i < mSize; ++i) {
            for (int j = 0; j < mSize; ++j)
//                mat[i][j] = 0;
                mat[i][j] = new BitSet(maxVecDim); // default value
        }

        SSACFG cfg = ir.getControlFlowGraph();
        BlockType[] blockTypes = new BlockType[cfg.getMaxNumber() + 1];
        for (int i = 0; i < blockTypes.length; ++i) {
            blockTypes[i] = BlockType.NORMAL;
        }
        getBlocksType(cfg, blockTypes);
        Iterator<ISSABasicBlock> blockIter = cfg.iterator();
        while (blockIter.hasNext()) {
            // control flow
            ISSABasicBlock block = blockIter.next();
            int blockNumber = block.getNumber();
            Iterator<ISSABasicBlock> succIter = cfg.getSuccNodes(block);
            while (succIter.hasNext()) {
                ISSABasicBlock succBlock = succIter.next();
                int succNumber = succBlock.getNumber();
                blockToBlcok(mat, varCount, blockNumber, succNumber, blockTypes);
            }

            // data flow
            Iterator<SSAInstruction> insIter = block.iterator();
            while (insIter.hasNext()) {
                SSAInstruction ins = insIter.next();
                if (ins != null) {
                    if (ins instanceof SSAGetInstruction) {
                        SSAGetInstruction getIns = (SSAGetInstruction) ins;
                        FieldReference fr = getIns.getDeclaredField();
                        int staticVar = fieldMap.get(fr.getName().toString());
                        int def = getIns.getDef();
                        variableToBlock(mat, varCount, staticVar, blockNumber, blockTypes);
                        variableToBlock(mat, varCount, def, blockNumber, blockTypes);

                        setDataFlow(mat[staticVar-1][def-1], 28, typeCode[def], typeCode[staticVar]);
                    } else if (ins instanceof SSAPutInstruction) {
                        SSAPutInstruction putIns = (SSAPutInstruction) ins;
                        FieldReference fr = putIns.getDeclaredField();
                        int staticVar = fieldMap.get(fr.getName().toString());
                        int var = putIns.getUse(0);
                        variableToBlock(mat, varCount, staticVar, blockNumber, blockTypes);
                        variableToBlock(mat, varCount, var, blockNumber, blockTypes);

                        setDataFlow(mat[var - 1][staticVar - 1], 38, typeCode[staticVar], typeCode[var]);
                    } else if (ins instanceof SSANewInstruction) {
                        SSANewInstruction newIns = (SSANewInstruction) ins;
                        int def = newIns.getDef();
                        variableToBlock(mat, varCount, def, blockNumber, blockTypes);
                        if (newIns.getNumberOfUses() == 0) {// single variable
                            setDataFlow(mat[def - 1][def - 1], 35, typeCode[def], typeCode[def]);
                        }
                        else { // array
                            int var = newIns.getUse(0);
                            variableToBlock(mat, varCount, var, blockNumber, blockTypes);
                            setDataFlow(mat[var - 1][def - 1], 35, typeCode[def], typeCode[var]);
                        }
                    } else if (ins instanceof SSAConversionInstruction) {
                        SSAConversionInstruction convIns = (SSAConversionInstruction) ins;
                        int def = convIns.getDef();
                        int var = convIns.getUse(0);
                        variableToBlock(mat, varCount, def, blockNumber, blockTypes);
                        variableToBlock(mat, varCount, var, blockNumber, blockTypes);

                        setDataFlow(mat[var - 1][def - 1], 26, typeCode[def], typeCode[var]);
                    } else if (ins instanceof SSAArrayLoadInstruction) {
                        SSAArrayLoadInstruction loadIns = (SSAArrayLoadInstruction) ins;
                        int def = loadIns.getDef();
                        int addr = loadIns.getArrayRef();
                        int index = loadIns.getIndex();
                        variableToBlock(mat, varCount, def, blockNumber, blockTypes);
                        variableToBlock(mat, varCount, addr, blockNumber, blockTypes);
                        variableToBlock(mat, varCount, index, blockNumber, blockTypes);

                        setDataFlow(mat[addr - 1][def - 1], 3, typeCode[def], typeCode[addr]);
                        setDataFlow(mat[index - 1][def - 1], 3, typeCode[def], typeCode[index]);
                    } else if (ins instanceof SSAArrayStoreInstruction) {
                        SSAArrayStoreInstruction storeIns = (SSAArrayStoreInstruction) ins;
                        int def = storeIns.getValue();
                        int addr = storeIns.getArrayRef();
                        int index = storeIns.getIndex();
                        variableToBlock(mat, varCount, def, blockNumber, blockTypes);
                        variableToBlock(mat, varCount, addr, blockNumber, blockTypes);
                        variableToBlock(mat, varCount, index, blockNumber, blockTypes);

                        setDataFlow(mat[addr - 1][def - 1], 4, typeCode[def], typeCode[addr]);
                        setDataFlow(mat[index - 1][def - 1], 4, typeCode[def], typeCode[index]);
                    } else if (ins instanceof SSAArrayLengthInstruction) {
                        SSAArrayLengthInstruction lengthIns = (SSAArrayLengthInstruction) ins;
                        int ref = lengthIns.getArrayRef();
                        int def = lengthIns.getDef();
                        variableToBlock(mat, varCount, def, blockNumber, blockTypes);
                        variableToBlock(mat, varCount, ref, blockNumber, blockTypes);

                        setDataFlow(mat[ref - 1][def - 1], 2, typeCode[def], typeCode[ref]);
                    } else if (ins instanceof SSAInvokeInstruction) {
                        SSAInvokeInstruction invokeIns = (SSAInvokeInstruction) ins;
                        int numUses = invokeIns.getNumberOfUses();
                        int numReturns = invokeIns.getNumberOfReturnValues();
                        int exception = invokeIns.getException();
                        variableToBlock(mat, varCount, exception, blockNumber, blockTypes);

                        //detect simple method invocation
                        String invokeStr = invokeIns.toString().toLowerCase();
                        if (invokeStr.contains("compareto") && (numUses == 2) && (numReturns == 1)) {
                            int def = invokeIns.getReturnValue(0);
                            int lvar = invokeIns.getUse(0);
                            int rvar = invokeIns.getUse(1);
                            variableToBlock(mat, varCount, lvar, blockNumber, blockTypes);
                            variableToBlock(mat, varCount, rvar, blockNumber, blockTypes);
                            variableToBlock(mat, varCount, def, blockNumber, blockTypes);

                            setDataFlow(mat[lvar - 1][def - 1], 19, typeCode[def], typeCode[lvar]);
                            setDataFlow(mat[rvar - 1][def - 1], 19, typeCode[def], typeCode[rvar]);

                            continue;
                        } else if (invokeStr.contains("equals") && (numUses == 2) && (numReturns == 1)) {
                            int def = invokeIns.getReturnValue(0);
                            int lvar = invokeIns.getUse(0);
                            int rvar = invokeIns.getUse(1);
                            variableToBlock(mat, varCount, lvar, blockNumber, blockTypes);
                            variableToBlock(mat, varCount, rvar, blockNumber, blockTypes);
                            variableToBlock(mat, varCount, def, blockNumber, blockTypes);

                            setDataFlow(mat[lvar - 1][def - 1], 17, typeCode[def], typeCode[lvar]);
                            setDataFlow(mat[rvar - 1][def - 1], 17, typeCode[def], typeCode[rvar]);

                            continue;
                        }

                        //relation between parameters themselves. [gang. 2016.09.26]
                        if (numUses > 1) {
                            for (int i = 0; i < numUses - 1; ++i) {
                                int param_i = invokeIns.getUse(i);
                                for (int j = i + 1; j < numUses; ++j) {
                                    int param_j = invokeIns.getUse(j);
                                    setDataFlow(mat[param_i - 1][param_j - 1], 31, typeCode[param_j], typeCode[param_i]);
                                    setDataFlow(mat[param_j - 1][param_i - 1], 31, typeCode[param_i], typeCode[param_j]);
                                }
                            }
                        }

                        //relation between params and return value
                        if (numReturns == 0) { // no return value
                            for (int k = 0; k < numUses; k++) {
                                int parameter = invokeIns.getUse(k);
                                variableToBlock(mat, varCount, parameter, blockNumber, blockTypes);
                                setDataFlow(mat[parameter - 1][exception - 1], 31, typeCode[exception], typeCode[parameter]);
                            }
                        } else {
                            int def = invokeIns.getReturnValue(0);
                            variableToBlock(mat, varCount, def, blockNumber, blockTypes);
                            setDataFlow(mat[def - 1][exception - 1], 31, typeCode[exception], typeCode[def]);
                            for (int k = 0; k < numUses; k++) {
                                int parameter = invokeIns.getUse(k);
                                variableToBlock(mat, varCount, parameter, blockNumber, blockTypes);
                                setDataFlow(mat[parameter - 1][def - 1], 31, typeCode[def], typeCode[parameter]);
                            }
                        }
                    } else if (ins instanceof SSAPiInstruction) {
                        SSAPiInstruction piIns = (SSAPiInstruction) ins;
                        int def = piIns.getDef();
                        int var = piIns.getUse(0);
                        variableToBlock(mat, varCount, def, blockNumber, blockTypes);
                        variableToBlock(mat, varCount, var, blockNumber, blockTypes);

                        setDataFlow(mat[var - 1][def - 1], 37, typeCode[def], typeCode[var]);
                    } else if (ins instanceof SSAUnaryOpInstruction && !(ins instanceof SSAPiInstruction)) {
                        SSAUnaryOpInstruction opIns = (SSAUnaryOpInstruction) ins;
                        String op = opIns.getOpcode().toString();
                        if (op != "neg")
                            System.out.println("Excluded op in UnaryOpInstruction: " + op);

                        int def = opIns.getDef();
                        int var = opIns.getUse(0);
                        variableToBlock(mat, varCount, def, blockNumber, blockTypes);
                        variableToBlock(mat, varCount, var, blockNumber, blockTypes);

                        setDataFlow(mat[var - 1][def - 1], 43, typeCode[def], typeCode[var]);
                    } else if (ins instanceof SSABinaryOpInstruction) {
                        SSABinaryOpInstruction opIns = (SSABinaryOpInstruction) ins;
                        String op = opIns.getOperator().toString();
                        int def = opIns.getDef();
                        int lvar = opIns.getUse(0);
                        int rvar = opIns.getUse(1);
                        variableToBlock(mat, varCount, def, blockNumber, blockTypes);
                        variableToBlock(mat, varCount, lvar, blockNumber, blockTypes);
                        variableToBlock(mat, varCount, rvar, blockNumber, blockTypes);

                        // get different value according to the op
                        switch (op) {
                            case "add":
                                setDataFlow(mat[rvar - 1][def - 1], 5, typeCode[def], typeCode[lvar]);
                                setDataFlow(mat[rvar - 1][def - 1], 5, typeCode[def], typeCode[rvar]);
                                break;
                            case "sub":
                                setDataFlow(mat[lvar - 1][def - 1], 6, typeCode[def], typeCode[lvar]);
                                setDataFlow(mat[rvar - 1][def - 1], 6, typeCode[def], typeCode[rvar]);
                                break;
                            case "mul":
                                setDataFlow(mat[lvar - 1][def - 1], 7, typeCode[def], typeCode[lvar]);
                                setDataFlow(mat[rvar - 1][def - 1], 7, typeCode[def], typeCode[rvar]);
                                break;
                            case "div":
                                setDataFlow(mat[lvar - 1][def - 1], 8, typeCode[def], typeCode[lvar]);
                                setDataFlow(mat[rvar - 1][def - 1], 8, typeCode[def], typeCode[rvar]);
                                break;
                            case "rem":
                                setDataFlow(mat[lvar - 1][def - 1], 9, typeCode[def], typeCode[lvar]);
                                setDataFlow(mat[rvar - 1][def - 1], 9, typeCode[def], typeCode[rvar]);
                                break;
                            case "and":
                                setDataFlow(mat[lvar - 1][def - 1], 10, typeCode[def], typeCode[lvar]);
                                setDataFlow(mat[rvar - 1][def - 1], 10, typeCode[def], typeCode[rvar]);
                                break;
                            case "or":
                                setDataFlow(mat[lvar - 1][def - 1], 11, typeCode[def], typeCode[lvar]);
                                setDataFlow(mat[rvar - 1][def - 1], 11, typeCode[def], typeCode[rvar]);
                                break;
                            case "xor":
                                setDataFlow(mat[lvar - 1][def - 1], 12, typeCode[def], typeCode[lvar]);
                                setDataFlow(mat[rvar - 1][def - 1], 12, typeCode[def], typeCode[rvar]);
                                break;
                            case "SHL":
                                setDataFlow(mat[lvar - 1][def - 1], 13, typeCode[def], typeCode[lvar]);
                                setDataFlow(mat[rvar - 1][def - 1], 13, typeCode[def], typeCode[rvar]);
                                break;
                            case "SHR":
                                setDataFlow(mat[lvar - 1][def - 1], 14, typeCode[def], typeCode[lvar]);
                                setDataFlow(mat[rvar - 1][def - 1], 14, typeCode[def], typeCode[rvar]);
                                break;
                            default:
                                setDataFlow(mat[lvar - 1][def - 1], 15, typeCode[def], typeCode[lvar]);
                                setDataFlow(mat[rvar - 1][def - 1], 15, typeCode[def], typeCode[rvar]);
                                System.out.println("Excluded op in BinaryOpInstruction: " + op);
                                break;
                        }
                    } else if (ins instanceof SSAComparisonInstruction) {
                        SSAComparisonInstruction compIns = (SSAComparisonInstruction) ins;
                        IComparisonInstruction.Operator op = compIns.getOperator();
                        int def = compIns.getDef();
                        int lvar = compIns.getUse(0);
                        int rvar = compIns.getUse(1);
                        variableToBlock(mat, varCount, def, blockNumber, blockTypes);
                        variableToBlock(mat, varCount, lvar, blockNumber, blockTypes);
                        variableToBlock(mat, varCount, rvar, blockNumber, blockTypes);

                        switch (op) {
                            case CMP:
                                setDataFlow(mat[lvar - 1][def - 1], 17, typeCode[def], typeCode[lvar]);
                                setDataFlow(mat[rvar - 1][def - 1], 17, typeCode[def], typeCode[rvar]);
                                break;
                            case CMPL:
                                setDataFlow(mat[lvar - 1][def - 1], 18, typeCode[def], typeCode[lvar]);
                                setDataFlow(mat[rvar - 1][def - 1], 18, typeCode[def], typeCode[rvar]);
                                break;
                            case CMPG:
                                setDataFlow(mat[lvar - 1][def - 1], 19, typeCode[def], typeCode[lvar]);
                                setDataFlow(mat[rvar - 1][def - 1], 19, typeCode[def], typeCode[rvar]);
                                break;
                        }
                    } else if (ins instanceof SSAConditionalBranchInstruction) {
                        SSAConditionalBranchInstruction condIns = (SSAConditionalBranchInstruction) ins;
                        IConditionalBranchInstruction.Operator op = (IConditionalBranchInstruction.Operator) condIns
                                .getOperator();
                        int lvar = condIns.getUse(0);
                        int rvar = condIns.getUse(1);
                        variableToBlock(mat, varCount, lvar, blockNumber, blockTypes);
                        variableToBlock(mat, varCount, rvar, blockNumber, blockTypes);

                        switch (op) {
                            case EQ:
                                setDataFlow(mat[rvar - 1][lvar - 1], 20, typeCode[lvar], typeCode[rvar]);
                                break;
                            case NE:
                                setDataFlow(mat[rvar - 1][lvar - 1], 21, typeCode[lvar], typeCode[rvar]);
                                break;
                            case LT:
                                setDataFlow(mat[rvar - 1][lvar - 1], 22, typeCode[lvar], typeCode[rvar]);
                                break;
                            case GE:
                                setDataFlow(mat[rvar - 1][lvar - 1], 23, typeCode[lvar], typeCode[rvar]);
                                break;
                            case GT:
                                setDataFlow(mat[rvar - 1][lvar - 1], 24, typeCode[lvar], typeCode[rvar]);
                                break;
                            case LE:
                                setDataFlow(mat[rvar - 1][lvar - 1], 25, typeCode[lvar], typeCode[rvar]);
                                break;
                        }
                    } else if (ins instanceof SSAPhiInstruction) {
                        SSAPhiInstruction phiIns = (SSAPhiInstruction) ins;
                        int def = phiIns.getDef();
                        int lvar = phiIns.getUse(0);
                        int rvar = phiIns.getUse(1);
                        variableToBlock(mat, varCount, def, blockNumber, blockTypes);
                        variableToBlock(mat, varCount, lvar, blockNumber, blockTypes);
                        variableToBlock(mat, varCount, rvar, blockNumber, blockTypes);

                        setDataFlow(mat[lvar - 1][def - 1], 36, typeCode[def], typeCode[lvar]);
                        setDataFlow(mat[rvar - 1][def - 1], 36, typeCode[def], typeCode[rvar]);
                    } else if (ins instanceof SSAGetCaughtExceptionInstruction) {
                        SSAGetCaughtExceptionInstruction caughtIns = (SSAGetCaughtExceptionInstruction) ins;
                        int var = caughtIns.getDef();
                        variableToBlock(mat, varCount, var, blockNumber, blockTypes);

                        setDataFlow(mat[var - 1][var - 1], 27, typeCode[var], typeCode[var]);
                    } else if (ins instanceof SSAThrowInstruction) {
                        SSAThrowInstruction throwIns = (SSAThrowInstruction) ins;
                        int var = throwIns.getUse(0);
                        variableToBlock(mat, varCount, var, blockNumber, blockTypes);

                        setDataFlow(mat[var - 1][var - 1], 42, typeCode[var], typeCode[var]);
                    } else if (ins instanceof SSACheckCastInstruction) {
                        SSACheckCastInstruction castIns = (SSACheckCastInstruction) ins;
                        int def = castIns.getDef();
                        int var = castIns.getUse(0);
                        variableToBlock(mat, varCount, def, blockNumber, blockTypes);
                        variableToBlock(mat, varCount, var, blockNumber, blockTypes);

                        setDataFlow(mat[var - 1][def - 1], 16, typeCode[def], typeCode[var]);
                    } else if (ins instanceof SSAInstanceofInstruction) {
                        SSAInstanceofInstruction instanceIns = (SSAInstanceofInstruction) ins;
                        int def = instanceIns.getDef();
                        int var = instanceIns.getUse(0);
                        variableToBlock(mat, varCount, def, blockNumber, blockTypes);
                        variableToBlock(mat, varCount, var, blockNumber, blockTypes);

                        setDataFlow(mat[var - 1][def - 1], 30, typeCode[def], typeCode[var]);
                    } else if (ins instanceof SSAReturnInstruction || ins instanceof SSAGotoInstruction
                            || ins instanceof SSASwitchInstruction) {
                        // related to control flow
                        // have nothing to do with data flow
//                        System.out.println("Switch Instruction");
                    } else {
                        excludedInstruction = excludedInstruction + ins.toString() + '\n';
                    }
                }
            }
        }


        String dir = base_dir + FilenameUtils.removeExtension(FilenameUtils.getName(appJar));
        File filedir = new File(dir);
        if (!filedir.exists()) {
            filedir.mkdir();
        }
        String funcName = mName.substring(mName.lastIndexOf('.') + 1);
        funcName = funcName.substring(0, funcName.indexOf('('));
        String parameterString = mName.substring(mName.indexOf('(') + 1, mName.indexOf(')'));
        if (parameterString.length() == 0)
            funcName += "()";
        else if (parameterString.contains(";")) {
            String[] parameters = parameterString.split(";");
            funcName += "(";
            for (int i = 0; i < parameters.length - 1; ++i) {
                String param = "";
                if (parameters[i].contains("/"))
                    param = parameters[i].substring(parameters[i].lastIndexOf('/') + 1);
                else {
                    if (parameters[i].matches("\\[?[IDBFSJ]+")) {
                        byte[] reg = parameters[i].getBytes();
                        int l = 0;
                        while (l < reg.length) {
                            if (reg[l] == '[') {
                                switch (reg[l + 1]) {
                                    case 'I': {
                                        param += "int[]";
                                        break;
                                    }
                                    case 'D': {
                                        param += "double[]";
                                        break;
                                    }
                                    case 'B': {
                                        param += "byte[]";
                                        break;
                                    }
                                    case 'J': {
                                        param += "long[]";
                                        break;
                                    }
                                    case 'F': {
                                        param += "float[]";
                                        break;
                                    }
                                    case 'S': {
                                        param += "short[]";
                                        break;
                                    }
                                    case 'Z': {
                                        param += "boolean[]";
                                        break;
                                    }
                                    default:
                                        param += "[]";
                                }
                                l += 2;
                                if (l <= reg.length - 1)
                                    funcName += ",";
                            } else {
                                switch (reg[l]) {
                                    case 'I': {
                                        param += "int";
                                        break;
                                    }
                                    case 'D': {
                                        param += "double";
                                        break;
                                    }
                                    case 'B': {
                                        funcName += "byte";
                                        break;
                                    }
                                    case 'J': {
                                        param += "long";
                                        break;
                                    }
                                    case 'F': {
                                        param += "float";
                                        break;
                                    }
                                    case 'S': {
                                        param += "short";
                                        break;
                                    }
                                    case 'Z': {
                                        param += "boolean";
                                        break;
                                    }
                                    default:
                                        param += "";
                                }
                                l += 1;
                                if (l <= reg.length - 1)
                                    param += ",";
                            }
                        }
                    } else {
                        param = parameters[i];
                    }
                }
                funcName += param;
                funcName += ',';
            }
            String param = "";
            if (parameters[parameters.length - 1].contains("/"))
                param = parameters[parameters.length - 1].substring(parameters[parameters.length - 1].lastIndexOf('/') + 1);
            else {
                if (parameters[parameters.length - 1].matches("\\[?[IDBFSJ]+")) {
                    byte[] reg = parameters[parameters.length - 1].getBytes();
                    int l = 0;
                    while (l < reg.length) {
                        if (reg[l] == '[') {
                            switch (reg[l + 1]) {
                                case 'I': {
                                    param += "int[]";
                                    break;
                                }
                                case 'D': {
                                    param += "double[]";
                                    break;
                                }
                                case 'B': {
                                    param += "byte[]";
                                    break;
                                }
                                case 'J': {
                                    param += "long[]";
                                    break;
                                }
                                case 'F': {
                                    param += "float[]";
                                    break;
                                }
                                case 'S': {
                                    param += "short[]";
                                    break;
                                }
                                case 'Z': {
                                    param += "boolean[]";
                                    break;
                                }
                                default:
                                    param += "[]";
                            }
                            l += 2;
                            if (l <= reg.length - 1)
                                funcName += ",";
                        } else {
                            switch (reg[l]) {
                                case 'I': {
                                    param += "int";
                                    break;
                                }
                                case 'D': {
                                    param += "double";
                                    break;
                                }
                                case 'B': {
                                    funcName += "byte";
                                    break;
                                }
                                case 'J': {
                                    param += "long";
                                    break;
                                }
                                case 'F': {
                                    param += "float";
                                    break;
                                }
                                case 'S': {
                                    param += "short";
                                    break;
                                }
                                case 'Z': {
                                    param += "boolean";
                                    break;
                                }
                                default:
                                    param += "";
                            }
                            l += 1;
                            if (l <= reg.length - 1)
                                param += ",";
                        }
                    }
                } else {
                    param = parameters[parameters.length - 1];
                }
            }
            funcName += param;
            funcName += ")";
        } else {
            funcName += "(";
            byte[] reg = parameterString.getBytes();
            int l = 0;
            while (l < reg.length) {
                if (reg[l] == '[') {
                    switch (reg[l + 1]) {
                        case 'I': {
                            funcName += "int[]";
                            break;
                        }
                        case 'D': {
                            funcName += "double[]";
                            break;
                        }
                        case 'B': {
                            funcName += "byte[]";
                            break;
                        }
                        case 'J': {
                            funcName += "long[]";
                            break;
                        }
                        case 'F': {
                            funcName += "float[]";
                            break;
                        }
                        case 'S': {
                            funcName += "short[]";
                            break;
                        }
                        case 'Z': {
                            funcName += "boolean[]";
                            break;
                        }
                        default:
                            funcName += "[]";
                    }
                    l += 2;
                    if (l <= reg.length - 1)
                        funcName += ",";
                } else {
                    switch (reg[l]) {
                        case 'I': {
                            funcName += "int";
                            break;
                        }
                        case 'D': {
                            funcName += "double";
                            break;
                        }
                        case 'B': {
                            funcName += "byte";
                            break;
                        }
                        case 'J': {
                            funcName += "long";
                            break;
                        }
                        case 'F': {
                            funcName += "float";
                            break;
                        }
                        case 'S': {
                            funcName += "short";
                            break;
                        }
                        case 'Z': {
                            funcName += "boolean";
                            break;
                        }
                        default:
                            funcName += "";
                    }
                    l += 1;
                    if (l <= reg.length - 1)
                        funcName += ",";
                }
            }
            funcName += ")";
        }
//        String fileName = mName + ".txt";
        String fileName = funcName + ".txt";
        String filePath = dir + "/" + className;
        try {
            File childDir = new File(filePath);
            if (!childDir.exists()) {
                childDir.mkdir();
            }
            File file = new File(filePath + "/" + fileName);
            if (file.exists())  //clear previous data
                file.delete();
            file.createNewFile();
            FileWriter out = new FileWriter(file, true);

            //record the actual varCount + blockCount at the last element of the mat
//            mat[fixedSize - 1][fixedSize - 1] = varCount + blockCount;
            for (int i = 0; i < fixedSize; ++i) {
                for (int j = 0; j < fixedSize; ++j)
                    out.write(mat[i][j].toString() + "\t");
                out.write("\n");
            }

            out.close();


        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    // record the relationship between variables and blocks
    public static void variableToBlock(BitSet[][] mat, int varCount, int var, int block, BlockType[] blockTypes) {
        int insType = blockWeights.get(blockTypes[block]);
        mat[var-1][varCount+block].set(insType);
        mat[varCount+block][var-1].set(insType);
    }

    public static void blockToBlcok(BitSet[][] mat, int varCount, int block, int succBlock, BlockType[] blockTypes) {
        int instType = blockWeights.get(blockTypes[block]);
        mat[varCount + block][varCount + succBlock].set(instType);
    }

    public static void setDataFlow(BitSet bs, int instType, int opCode1, int opCode2) {
        bs.set(instType + 37);
        setOpCode(bs, opCode1, 0);
        setOpCode(bs, opCode2, 18);
    }

    public static void setOpCode(BitSet bs, int opCode, int offset) {
        int idx = opCode & 0x03;
        bs.set(offset+idx);
        idx = (opCode & 0x0C) >> 2;
        bs.set(offset+idx);
        idx = (opCode & 0xF0) >> 4;
        bs.set(offset+idx);
    }
}

/*
BooleanName Z 	1
ByteName B 		2
CharName C 		3
DoubleName D 	4
FloatName F 	5
IntName I 		6
LongName J 		7
ShortName S 	8
VoidName V 		9
OtherPrimi P 	10
ClassTypeCode L
Java 			11
User-defined 	12


ArrayTypeCode = '[';
PointerTypeCode = '*';
ReferenceTypeCode = '&';

SSAAddressOfInstruction 1 			XX
SSAArrayLengthInstruction 2
SSAArrayLoadInstruction 3
SSAArrayStoreInstruction 4
SSABinaryOpInstruction 5~15
SSACheckCastInstruction 16
SSAComparisonInstruction 17~19
SSAConditionalBranchInstruction 20~25
SSAConversionInstruction 26
SSAGetCaughtExceptionInstruction 27
SSAGetInstruction 28
SSAGotoInstruction 29
SSAInstanceofInstruction 30
SSAInvokeInstruction 31
SSALoadIndirectInstruction 32 		XX
SSALoadMetadataInstruction 33 		XX
SSAMonitorInstruction 34 			XX
SSANewInstruction 35
SSAPhiInstruction 36
SSAPiInstruction 37
SSAPutInstruction 38
SSAReturnInstruction 39
SSAStoreIndirectInstruction 40 		XX
SSASwitchInstruction 41
SSAThrowInstruction 42
SSAUnaryOpInstruction 43


x = [op1, op2, InstType, BBType]
op1: 0-18        0-2 modifier property, 3-6 var type, 7-18: var data type
op2: 19-37 offset:18
Instype: 38-80 offset:37
BBtype: 81-87

Dim of x = 19+19+43+7 = 88

*/
