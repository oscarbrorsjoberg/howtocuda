#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>
#include <string>
#include <map>
#include <tuple>

/* #include "glMaxValues.hpp" */
#include "imguiBackend.hpp"
#include "glfw_state.hpp"


static std::map<std::string, unsigned int> max_values;

void initMaxValues(){
#define INSERT_ELEMENT(p) \
	[&] { \
		GLint value; \
		glGetIntegerv(p, &value); \
		max_values[#p] = value; \
	}() 
	INSERT_ELEMENT(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS);
	INSERT_ELEMENT(GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS);
	INSERT_ELEMENT(GL_MAX_COMPUTE_UNIFORM_BLOCKS);
	INSERT_ELEMENT(GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS);
	INSERT_ELEMENT(GL_MAX_COMPUTE_UNIFORM_COMPONENTS);
	INSERT_ELEMENT(GL_MAX_COMPUTE_ATOMIC_COUNTERS);
	INSERT_ELEMENT(GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS);
	INSERT_ELEMENT(GL_MAX_COMBINED_COMPUTE_UNIFORM_COMPONENTS);
	INSERT_ELEMENT(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS);
	INSERT_ELEMENT(GL_MAX_PATCH_VERTICES);
	/* INSERT_ELEMENT(GL_MAX_COMPUTE_WORK_GROUP_COUNT); */
	/* INSERT_ELEMENT(GL_MAX_COMPUTE_WORK_GROUP_SIZE); */
	INSERT_ELEMENT(GL_MAX_DEBUG_GROUP_STACK_DEPTH);
	/* INSERT_ELEMENT(GL_MAX_3D_TEXTURE_SIZE); */
	INSERT_ELEMENT(GL_MAX_ARRAY_TEXTURE_LAYERS);
	INSERT_ELEMENT(GL_MAX_CLIP_DISTANCES);
	INSERT_ELEMENT(GL_MAX_COLOR_TEXTURE_SAMPLES);
	INSERT_ELEMENT(GL_MAX_COMBINED_ATOMIC_COUNTERS);
	INSERT_ELEMENT(GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS);
	INSERT_ELEMENT(GL_MAX_COMBINED_GEOMETRY_UNIFORM_COMPONENTS);
	INSERT_ELEMENT(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS);
	INSERT_ELEMENT(GL_MAX_COMBINED_UNIFORM_BLOCKS);
	INSERT_ELEMENT(GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS);
	/* INSERT_ELEMENT(GL_MAX_CUBE_MAP_TEXTURE_SIZE); */
	INSERT_ELEMENT(GL_MAX_DEPTH_TEXTURE_SAMPLES);
	INSERT_ELEMENT(GL_MAX_DRAW_BUFFERS);
	INSERT_ELEMENT(GL_MAX_DUAL_SOURCE_DRAW_BUFFERS);
	INSERT_ELEMENT(GL_MAX_ELEMENTS_INDICES);
	INSERT_ELEMENT(GL_MAX_ELEMENTS_VERTICES);
	INSERT_ELEMENT(GL_MAX_FRAGMENT_ATOMIC_COUNTERS);
	INSERT_ELEMENT(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS);
	INSERT_ELEMENT(GL_MAX_FRAGMENT_INPUT_COMPONENTS);
	INSERT_ELEMENT(GL_MAX_FRAGMENT_UNIFORM_COMPONENTS);
	INSERT_ELEMENT(GL_MAX_FRAGMENT_UNIFORM_VECTORS);
	INSERT_ELEMENT(GL_MAX_FRAGMENT_UNIFORM_BLOCKS);
	INSERT_ELEMENT(GL_MAX_FRAMEBUFFER_WIDTH);
	INSERT_ELEMENT(GL_MAX_FRAMEBUFFER_HEIGHT);
	INSERT_ELEMENT(GL_MAX_FRAMEBUFFER_LAYERS);
	INSERT_ELEMENT(GL_MAX_FRAMEBUFFER_SAMPLES);
	INSERT_ELEMENT(GL_MAX_GEOMETRY_ATOMIC_COUNTERS);
	INSERT_ELEMENT(GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS);
	INSERT_ELEMENT(GL_MAX_GEOMETRY_INPUT_COMPONENTS);
	INSERT_ELEMENT(GL_MAX_GEOMETRY_OUTPUT_COMPONENTS);
	INSERT_ELEMENT(GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS);
	INSERT_ELEMENT(GL_MAX_GEOMETRY_UNIFORM_BLOCKS);
	INSERT_ELEMENT(GL_MAX_GEOMETRY_UNIFORM_COMPONENTS);
	INSERT_ELEMENT(GL_MAX_INTEGER_SAMPLES);
	INSERT_ELEMENT(GL_MAX_LABEL_LENGTH);
	INSERT_ELEMENT(GL_MAX_PROGRAM_TEXEL_OFFSET);
	/* INSERT_ELEMENT(/1* GL_MAX_RECTANGLE_TEXTURE_SIZE, *); */
	/* INSERT_ELEMENT(/1* GL_MAX_RENDERBUFFER_SIZE, *); */
	INSERT_ELEMENT(GL_MAX_SAMPLE_MASK_WORDS);
	INSERT_ELEMENT(GL_MAX_SERVER_WAIT_TIMEOUT);
	INSERT_ELEMENT(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS);
	INSERT_ELEMENT(GL_MAX_TESS_CONTROL_ATOMIC_COUNTERS);
	INSERT_ELEMENT(GL_MAX_TESS_EVALUATION_ATOMIC_COUNTERS);
	INSERT_ELEMENT(GL_MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS);
	INSERT_ELEMENT(GL_MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS);
	/* INSERT_ELEMENT(/1* GL_MAX_TEXTURE_BUFFER_SIZE, *); */
	INSERT_ELEMENT(	GL_MAX_TEXTURE_IMAGE_UNITS);
	INSERT_ELEMENT(GL_MAX_TEXTURE_LOD_BIAS);
	/* GL_MAX_TEXTURE_SIZE, */
	INSERT_ELEMENT(GL_MAX_UNIFORM_BUFFER_BINDINGS);
	/* INSERT_ELEMENT(GL_MAX_UNIFORM_BLOCK_SIZE); */
	INSERT_ELEMENT(GL_MAX_UNIFORM_LOCATIONS);
	INSERT_ELEMENT(GL_MAX_VARYING_COMPONENTS);
	INSERT_ELEMENT(GL_MAX_VARYING_VECTORS);
	INSERT_ELEMENT(GL_MAX_VARYING_FLOATS);
	INSERT_ELEMENT(GL_MAX_VERTEX_ATOMIC_COUNTERS);
	INSERT_ELEMENT(GL_MAX_VERTEX_ATTRIBS);
	INSERT_ELEMENT(GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS);
	INSERT_ELEMENT(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS);
	INSERT_ELEMENT(GL_MAX_VERTEX_UNIFORM_COMPONENTS);
	INSERT_ELEMENT(GL_MAX_VERTEX_UNIFORM_VECTORS);
	INSERT_ELEMENT(GL_MAX_VERTEX_OUTPUT_COMPONENTS);
	INSERT_ELEMENT(GL_MAX_VERTEX_UNIFORM_BLOCKS);
	/* INSERT_ELEMENT(GL_MAX_VIEWPORT_DIMS); */
	INSERT_ELEMENT(GL_MAX_VIEWPORTS);
	INSERT_ELEMENT(GL_MAX_VERTEX_ATTRIB_RELATIVE_OFFSET);
	INSERT_ELEMENT(GL_MAX_VERTEX_ATTRIB_BINDINGS);
	INSERT_ELEMENT(GL_MAX_ELEMENT_INDEX);
#undef INSERT_ELEMENT

}

// ----- ELEMENT CONTAINER
void ElementContainer::addElement(const std::string &name, Element* element)
{ 
	elements_.emplace(name, element);
}


void SideBar::setTexId(GLuint *texId, 
				unsigned int w, unsigned int h)

{
	(void)texId;
	(void)w;
	(void)h;
	return;
} 

// ----- SIDE BAR
void SideBar::set()
{

	ImGui::Begin(name_.c_str(), nullptr, ImGuiWindowFlags_NoResize | 
			ImGuiWindowFlags_NoMove);
	ImGui::SetWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - sizeX_, 0));
	ImGui::SetWindowSize(ImVec2(sizeX_, ImGui::GetIO().DisplaySize.y));
}

void SideBar::draw()
{
	set();

	for(auto& el: elements_){
		el.second->takeAction(el.first);
	}


	ImGui::Text("App average %.3f ms", 
			1000.0f / ImGui::GetIO().Framerate);
	ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);

	ImGui::End();    
}

// ----- WINDOW

void Window::set()
{
	ImGui::SetWindowSize(ImVec2(sizeX_, sizeY_));
}

void Window::setTexId(GLuint *texId, 
		unsigned int w, unsigned int h) 
{
	id_ = texId;
	imageWidth_ = w;
	imageHeight_ = h;
} 

void Window::draw()
{
	ImGui::Begin(name_.c_str(), NULL, 0);

	if(id_ ){
		ImGui::Image(reinterpret_cast<void*>(*id_), 
				ImVec2(imageWidth_, imageHeight_), ImVec2(0,1), ImVec2(1,0), 
				ImVec4(1.0f,1.0f,1.0f,1.0f), // color picker (1 for active ?)
				ImVec4(0.0f,0.0f,0.0f,0.0f) // border
				);
	}


	/* bool t = true; */
	/* ImGui::ShowMetricsWindow(&t); */

	/* for(auto& el: elements_){ */
	/* 	el.second->takeAction(el.first); */
	/* } */

	/* for(auto& item : max_values){ */
	/* 	ImGui::Text("%s: %d", item.first.c_str(), item.second); */
	/* } */

	ImGui::End();    
}


// ----- ELEMENTS

// ----- SLIDER

template<>
void Slider<int>::takeAction(std::string _name)
{
	ImGui::SliderInt(_name.c_str(), _value.get(), _min, _max);     
}

template<>
void Slider<float>::takeAction(std::string _name)
{
	ImGui::SliderFloat(_name.c_str(), _value.get(), _min, _max);     
}

// ----- COLOR PICKER
template<>
void ColorPicker<glm::vec4>::takeAction(std::string _name)
{
	ImGui::ColorEdit3(_name.c_str(), (float*)(_color.get()));     
}

// ----- BUTTON
void Button::takeAction(std::string _name)
{
	(*_pressed.get()) = ImGui::Button(_name.c_str());
}
// --- PopUpSelect

template<>
void PopUpSelect<std::string>::takeAction(std::string _name)
{
	if (ImGui::Button(_name.c_str()))
		ImGui::OpenPopup("popup");
	ImGui::SameLine();
	ImGui::TextUnformatted(selected_.get()->c_str());
	if (ImGui::BeginPopup("popup"))
	{
		ImGui::Text("Selections");
		ImGui::Separator();
		for (auto &sel: selections_){
			if (ImGui::Selectable(sel.c_str())){
				(*selected_.get()) = sel;
			}
		}
		ImGui::EndPopup();
	}
}




// ----- UI BACKEND
//
void UiBackend::linkGlfwGL3(GLFWwindow *window, const char* glsl_version)
{

	initMaxValues(); // TODO: move this to own element

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	// Setup Platform/Renderer bindings
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();

	ImGui::NewFrame();

	/**/

	for(auto & [key, elCont]: elContainers_){
		elCont->set();
	}

	ImGui::End();    
	ImGui::EndFrame();
}

void UiBackend::draw()
{

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();

	ImGui::NewFrame();

	for(auto & [key, elCont]: elContainers_){
		elCont->draw();
	}

	// Render dear imgui to screen
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());   

}

std::tuple<unsigned int, unsigned int> 
UiBackend::getContainerSize(const std::string &key){

	if(elContainers_.count(key)){
		return std::make_tuple(
				elContainers_[key]->sizeX(),
				elContainers_[key]->sizeY()
				);
	}
	else{
		throw std::runtime_error("Element container does not contain " + key );
	}
}


void UiBackend::addElementContainer(const std::string &key, ElementContainer *elCont)
{
	if(!elContainers_.count(key)){
		elContainers_[key] = elCont;
	}
	else{
		throw std::runtime_error(key + " already in container");
	}
}

void UiBackend::addElement(const std::string &key, 
		const std::string &elementName, Element *element)
{
	if(elContainers_.count(key)){
		elContainers_[key]->addElement(elementName, element);
	}
	else{
		throw std::runtime_error("Element container does not contain " + key );
	}
}

void UiBackend::addImage(const std::string &key, GLuint *tid,
		unsigned int w, unsigned int h)
{
	if(elContainers_.count(key)){
		elContainers_[key]->setTexId(tid, w, h);
	}
	else{
		throw std::runtime_error("Element container does not contain " + key );
	}
}
