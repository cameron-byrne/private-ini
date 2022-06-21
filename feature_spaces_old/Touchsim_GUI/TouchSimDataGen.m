function varargout = TouchSimDataGen(varargin)
% TOUCHSIMDATAGEN MATLAB code for TouchSimDataGen.fig
%      TOUCHSIMDATAGEN, by itself, creates a new TOUCHSIMDATAGEN or raises the existing
%      singleton*.
%
%      H = TOUCHSIMDATAGEN returns the handle to a new TOUCHSIMDATAGEN or the handle to
%      the existing singleton*.
%
%      TOUCHSIMDATAGEN('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TOUCHSIMDATAGEN.M with the given input arguments.
%
%      TOUCHSIMDATAGEN('Property','Value',...) creates a new TOUCHSIMDATAGEN or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before TouchSimDataGen_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to TouchSimDataGen_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help TouchSimDataGen

% Last Modified by GUIDE v2.5 14-Jun-2020 16:26:27

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @TouchSimDataGen_OpeningFcn, ...
                   'gui_OutputFcn',  @TouchSimDataGen_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before TouchSimDataGen is made visible.
function TouchSimDataGen_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to TouchSimDataGen (see VARARGIN)

% Choose default command line output for TouchSimDataGen
handles.output = hObject;

% this will incluide all the directories to run the TouchSim model
setup_path;

handles.FiringArray=[];

load AR.mat;

handles.AR = AR;

axes(handles.axes1);

IndexFront = find(AR.location(:,1)<14);
IndexMid   = find(AR.location(:,1)>14 & AR.location(:,1)<40.45);
IndexBack  = find(AR.location(:,1)>40.45);


h_Mid  =plot(AR.location(IndexMid,2),AR.location(IndexMid,1),'.');
hold on
h_Back =plot(AR.location(IndexBack,2),AR.location(IndexBack,1),'.');
h_Front    = plot(AR.location(IndexFront,2),AR.location(IndexFront,1),'.');
handles.PlotRA = plot(0,0,'marker','.','color','g','markersize',20);
handles.PlotSA = plot(0,0,'marker','.','color','m','markersize',20);
handles.PlotPC = plot(0,0,'marker','.','color','k','markersize',20);
handles.StimLoc = plot(0,0,'marker','*','color','k','markersize',10);


legend('Mid','Back','Front','RA','SA','PC','Stim');

set(handles.axes1,'fontweight','bold','fontsize',12,'yDir','reverse')



axes(handles.axes4)

load fiber_type
AllLoc=[];
for Sel=1:10
    for Counter=1:fascicles{Sel}.nAxons
        AllLoc{Sel}(Counter,:) = fascicles{Sel}.axons{Counter}.location;
    end
end

hold on
for Sel=1:10
    plot(AllLoc{Sel}(:,1),AllLoc{Sel}(:,2),'Marker','.','color',[0.8 0.8 0.8],'linestyle','none')
end

xlim([-2.6 2])


load Mapping_New;

AllLoc = [];

IndexFront = [];
IndexMid    = [];
IndexEnd    = [];

for Sel = 1:1704
    %strcmp(Mapping_New(Sel).Placement,'Distal')
    AllLoc(Sel,:) = [Mapping_New(Sel).XPosition_s109 Mapping_New(Sel).YPosition_s109];
    if strcmp(Mapping_New(Sel).Placement,'Distal')
        IndexFront = [IndexFront Sel];
    else
        if strcmp(Mapping_New(Sel).Placement,'Middle')
            IndexMid    = [IndexMid Sel];
        else

            IndexEnd    = [IndexEnd Sel]; 
        end
        
    end
    
end


handles.ElecLocation = AllLoc;

plot(AllLoc(IndexMid,1),AllLoc(IndexMid,2),'Marker','.','color',h_Mid.Color,'linestyle','none');
hold on
plot(AllLoc(IndexBack,1),AllLoc(IndexBack,2),'Marker','.','color',h_Back.Color,'linestyle','none');
plot(AllLoc(IndexFront,1),AllLoc(IndexFront,2),'Marker','.','color',h_Front.Color,'linestyle','none');

handles.PlotRA1 = plot(0,0,'marker','.','color','g','markersize',20);
handles.PlotSA1= plot(0,0,'marker','.','color','m','markersize',20);
handles.PlotPC1 = plot(0,0,'marker','.','color','k','markersize',20);

set(handles.axes4,'xticklabel','','yticklabel','')

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes TouchSimDataGen wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = TouchSimDataGen_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function EditLocation_Callback(hObject, eventdata, handles)
% hObject    handle to EditLocation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of EditLocation as text
%        str2double(get(hObject,'String')) returns contents of EditLocation as a double

Temp = str2num(get(hObject,'string'));

set(handles.StimLoc,'yData',Temp(1),'xData',Temp(2));


% --- Executes on button press in PushButtonGenData.
function PushButtonGenData_Callback(hObject, eventdata, handles)
% hObject    handle to PushButtonGenData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Here is a script to generate a sample data
amp = str2num(get(handles.EditAmp,'string')); %Amplitude in mm (default 1)
len = 1; %total duraiton of stimulus in s (default 1)
loc = str2num(get(handles.EditLocation,'string')); %stimulus location in mm (default [0 0])
samp_freq = 5000; %sampling frequency in Hz, default 5000
ramp_len = str2num(get(handles.EditRampDuration,'string')); %duration of on and off ramps in s (default 0.05)
ramp_type = 'lin'; %ramp type (default lin, can be either 'lin' or 'sine')
pin_size = str2num(get(handles.EditProbeSize,'string'));% 3; %probe radius in mm
pre_indent = 0; %static indentation throughout trial (depth)

% Stimulus generation:
handles.S = stim_ramp(amp,len,loc,samp_freq,ramp_len,ramp_type,pin_size,pre_indent);

axes(handles.axes3);

plot([1:5000]/5000,handles.S.trace,'linewidth',2);

set(handles.axes3,'fontweight','bold');
title('Stim Signal','fontsize',14);
xlabel('Time (sec)','fontweight','bold','fontsize',14);
ylabel('Stim Amplitude','fontweight','bold','fontsize',14);

load AR;

handles.r = AR.response(handles.S);
%%
guidata(hObject,handles);

axes(handles.axes2)
% plot the response
plot(handles.r)
xlabel('Time (sec)','fontweight','bold','fontsize',14)
title('Mechnical Model Response to Stim Signal','fontweight','bold','fontsize',14)

set(handles.axes2,'fontweight','bold')

linkaxes([handles.axes2 handles.axes3],'x');

%%

% Get thee firing array
FiringArray = zeros(1000,1704);

for Sel = 1:1704 % Select your axon from 1 to 1704
  
    % The firing of selected axon has been stored in the following:
    % This is exact time of the firing of the seclted axon for the entire during
    % of stimulation
    handles.r.responses(Sel).spikes;
    
    % We can convert these time to 1000 samples per second by rounding as:
    TempResponse =round(handles.r.responses(Sel).spikes*1000)/1000;
    % this is exact time of firing of selcted axon per millisecond
    
    % Find the index of firing for the selected axon
    IndexFiring = TempResponse*1000+1;
    
    if ~isempty(IndexFiring)
        FiringArray(IndexFiring,Sel) = 1;
    end
end

handles.FiringArray = FiringArray;

guidata(hObject,handles);


% --- Executes on mouse motion over figure - except title and menu.
function figure1_WindowButtonMotionFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
   



% get the location of click
Loc = get(handles.axes2,'CurrentPoint');

% get the limit of the current axeses
xLim = get(handles.axes2,'xLim');
yLim = get(handles.axes2,'yLim');


if Loc(1)> xLim(1) & Loc(1)<xLim(2) & Loc(3)>yLim(1) & Loc(3)<yLim(2) & ~isempty(handles.FiringArray)
    
    Index = fix(Loc(1)*1000);
    
    TimeBefore = str2num(get(handles.EditBefore,'string'));
    TimeAfter  = str2num(get(handles.EditAfter ,'string'));
    
    
    if Index < (TimeBefore+1)
        Index = TimeBefore+1;
    end
    
     if Index > 1000-TimeAfter
        Index = 1000-TimeAfter;
    end
    
    
    
    Temp = sum(handles.FiringArray([-TimeBefore:TimeAfter]+Index,:),1);
    
    IndexRA = find(Temp .* handles.AR.iRA);
    IndexSA = find(Temp .* handles.AR.iSA1);
    IndexPC = find(Temp .* handles.AR.iPC);
    
    set(handles.PlotRA,'yData',handles.AR.location(IndexRA,1),'xData',handles.AR.location(IndexRA,2));
    set(handles.PlotSA,'yData',handles.AR.location(IndexSA,1),'xData',handles.AR.location(IndexSA,2));
    set(handles.PlotPC,'yData',handles.AR.location(IndexPC,1),'xData',handles.AR.location(IndexPC,2));
    
    set(handles.PlotRA1,'yData',handles.ElecLocation(IndexRA,2),'xData',handles.ElecLocation(IndexRA,1));
    set(handles.PlotSA1,'yData',handles.ElecLocation(IndexSA,2),'xData',handles.ElecLocation(IndexSA,1));
    set(handles.PlotPC1,'yData',handles.ElecLocation(IndexPC,2),'xData',handles.ElecLocation(IndexPC,1));
    

    
    set(handles.TextPointer,'string',['Pointer : ' num2str(Index) ' mSec']);
  
end


% --- Executes on button press in CheckBoxRA.
function CheckBoxRA_Callback(hObject, eventdata, handles)
% hObject    handle to CheckBoxRA (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of CheckBoxRA

if get(hObject,'value')
    set(handles.PlotRA,'Visible','on');
    set(handles.PlotRA1,'Visible','on');
else
    set(handles.PlotRA,'Visible','off');
    set(handles.PlotRA1,'Visible','off');
end


% --- Executes on button press in CheckBoxSA.
function CheckBoxSA_Callback(hObject, eventdata, handles)
% hObject    handle to CheckBoxSA (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of CheckBoxSA


if get(hObject,'value')
    set(handles.PlotSA,'Visible','on');
    set(handles.PlotSA1,'Visible','on');
else
    set(handles.PlotSA,'Visible','off');
    set(handles.PlotSA1,'Visible','off');
end


% --- Executes on button press in CheckBoxPC.
function CheckBoxPC_Callback(hObject, eventdata, handles)
% hObject    handle to CheckBoxPC (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of CheckBoxPC


if get(hObject,'value')
    set(handles.PlotPC,'Visible','on');
    set(handles.PlotPC1,'Visible','on');
else
    set(handles.PlotPC,'Visible','off');
    set(handles.PlotPC1,'Visible','off');
end
