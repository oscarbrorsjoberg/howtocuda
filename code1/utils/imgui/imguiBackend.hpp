#ifndef IMGUIBACKEND_HPP_7TIIBXJ2
#define IMGUIBACKEND_HPP_7TIIBXJ2


#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"
#include <string>
#include <unordered_map>
#include <memory>
#include <cassert>
#include <vector>
#include <variant>
#include <tuple>

#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

class Element
{
  public:
    Element() = default;
    virtual void takeAction(std::string name) = 0;
    Element(const Element&) = delete; //No copy constructor.
    Element &operator=(const Element&) = delete; //No copy-assignment.
    virtual ~Element() = default;

  private:

};

template<typename T1>
class Slider : public Element
{
  public:
    Slider(std::shared_ptr<T1> valLink, const T1& min, const T1& max):
      _value(valLink)
      , _max(max)
      ,_min(min)
  {
  };

    Slider(const Slider&) = delete; // copy constructor.
    Slider &operator=(const Slider&) = delete; //No copy-assignment.

    // move assignement
    Slider &operator=(Slider&& ohter) = delete;
    // move custructor
    Slider(Slider &&other):
      _value(other._value)
    , _max(other.max)
    ,  _min(other.min)
    {};


    ~Slider() override = default;

    void takeAction(std::string name) override;

  private:
    std::shared_ptr<T1> _value;
    T1 _max;
    T1 _min;

};

template<typename T2>
class ColorPicker : public Element
{
  public:
    ColorPicker(std::shared_ptr<T2> valLink):
       _color(valLink)
  {};

    ColorPicker(const ColorPicker&) = delete; // copy constructor.
    ColorPicker &operator=(const ColorPicker&) = delete; //No copy-assignment.

    // move assignement
    ColorPicker &operator=(ColorPicker&& ohter) = delete;
    // move custructor
    ColorPicker(ColorPicker &&other):
     _color(other.color)
    {};

    ~ColorPicker() override = default;
    void takeAction(std::string name) override; 

    private:
    std::shared_ptr<T2> _color;

};


class Button : public Element
{
  public:
    Button(std::shared_ptr<bool> in):
      _pressed(in)
  {};

    Button(const Button&) = delete; // copy constructor.
    Button &operator=(const Button&) = delete; //No copy-assignment.

    // move assignement
    Button &operator=(Button&& ohter) = delete;
    // move custructor
    Button(Button &&other):
     _pressed(other._pressed)
    {};

    ~Button() override = default;
    void takeAction(std::string name) override; 

    private:
    std::shared_ptr<bool> _pressed;

};

template<typename T3>
class PopUpSelect : public Element
{
  public:
    PopUpSelect(std::vector<T3> selections, std::shared_ptr<T3> in):
			selections_((assert(selections.size() > 0), selections)),
			selected_(in)

  {
		(*selected_.get()) = selections.at(0);
	};

    PopUpSelect(const PopUpSelect&) = delete; // copy constructor.
    PopUpSelect &operator=(const PopUpSelect&) = delete; //No copy-assignment.

    // move assignement
    PopUpSelect &operator=(PopUpSelect&& ohter) = delete;
    // move custructor
    PopUpSelect(PopUpSelect &&other):
     selections_(other.selections_)
     , selected_(selections_.at(0))
    {};

    ~PopUpSelect() override = default;
    void takeAction(std::string name) override; 

    private:
    std::vector<T3> selections_;
		std::shared_ptr<T3> selected_;

};


class ElementContainer
{
	public:
		ElementContainer(const std::string &name, unsigned int sx, unsigned int sy):
			sizeX_{sx}
		, sizeY_{sy}
		, name_{name}
		{}

		ElementContainer() = delete;

		virtual void draw() = 0;
		virtual void set() = 0;
		virtual void setTexId(GLuint *tid, 
				unsigned int w, unsigned int h) = 0;

		int sizeX() const{return sizeX_;} 
		int sizeY() const {return sizeY_;}
		std::string name() const {return name_;}

    virtual ~ElementContainer()
    {
      for(auto& el: elements_)
        delete el.second;
    }

		// delete copy and move assignment and const
		ElementContainer(ElementContainer &&other) = delete;
		ElementContainer &operator=(ElementContainer &&other) = delete;

		ElementContainer(const ElementContainer&) = delete;
		ElementContainer &operator=(const ElementContainer&) = delete;


		void addElement(const std::string &name, Element *element);

	protected:
		unsigned int sizeX_;
		unsigned int sizeY_;

    std::unordered_map<std::string, Element*> elements_;
    std::string name_;

};




class Window : public ElementContainer
{

	private:
		GLuint *id_;
		unsigned int imageWidth_;
		unsigned int imageHeight_;

	public:
    Window(const std::string &name, int sx, int sy):
			ElementContainer(name, sx, sy),
			id_(nullptr),
			imageWidth_(0),
			imageHeight_(0)
    {}

    void draw() override;
    void set() override;

		void setTexId(GLuint *texId, 
				unsigned int w, unsigned int h) 
			override;


};

class SideBar : public ElementContainer
{

  public:
    SideBar(const std::string &name, int sx, int sy):
			ElementContainer(name, sx, sy)
    {}

    void draw() override;
    void set() override;

		void setTexId(GLuint *texId, 
				unsigned int w, unsigned int h) 
			override;



};


class UiBackend
{
	public:

		UiBackend() = default;  

		~UiBackend()
		{
			// free elConts
			for(auto & [key, elCont]: elContainers_)
				delete elCont;

			ImGui_ImplOpenGL3_Shutdown();
			ImGui::DestroyContext();
		};

		UiBackend(const UiBackend&) = delete; //No copy constructor.
		UiBackend &operator=(const UiBackend&) = delete; //No copy-assignment.

		void linkGlfwGL3(GLFWwindow *window, const char* glsl_version);
		void draw();

		void addElementContainer(const std::string &key, ElementContainer *elCont);
		// Adds element to speciefied element container
		void addElement(const std::string &key, 
				const std::string &elementName, Element *element);

		void addImage(const std::string &key, GLuint *tid,
				unsigned int w, unsigned int h);

		std::tuple<unsigned int, unsigned int> getContainerSize(const std::string &key);

				/* void setSideBar(const std::string &name, int sx, int sy); */
				/* void setWindow(const std::string &name, int sx, int sy); */

	private:
		std::unordered_map<std::string, ElementContainer*> elContainers_;
};

#endif /* end of include guard: IMGUIBACKEND_HPP_7TIIBXJ2 */
